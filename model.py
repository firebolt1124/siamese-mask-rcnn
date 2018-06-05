# Simaese Mask R-CNN Model

import tensorflow as tf
import sys
import os
import re
import time
import random
import numpy as np
import skimage.io
import imgaug

import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import multiprocessing

MASK_RCNN_MODEL_PATH = '/gpfs01/bethge/home/cmichaelis/tf-models/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize  

import utils as siamese_utils


def build_resnet_model(config):
    # Define model to run resnet twice for image and target
    # Define input image
    input_image = KL.Input(shape=[None,None,3],
                          name="input_image")
    # Compute ResNet activations
    C1, C2, C3, C4, C5 = modellib.resnet_graph(input_image, config.BACKBONE,
                                         stage5=True, train_bn=config.TRAIN_BN)
    # Return model
    return KM.Model([input_image], [C1, C2, C3, C4, C5], name="resnet_model")


def build_fpn_model(feature_maps=128):
    # Define model to run resnet+fpn twice for image and target
    # Define input image
    C2 = KL.Input(shape=[None,None,256], name="input_C2")
    C3 = KL.Input(shape=[None,None,512], name="input_C3")
    C4 = KL.Input(shape=[None,None,1024], name="input_C4")
    C5 = KL.Input(shape=[None,None,2048], name="input_C5")
    # Compute fpn activations
    P2, P3, P4, P5, P6 = fpn_graph(C2, C3, C4, C5, feature_maps=feature_maps)
    # Return model
    return KM.Model([C2, C3, C4, C5], [P2, P3, P4, P5, P6], name="fpn_model")


def fpn_graph(C2, C3, C4, C5, feature_maps=128):
    # CHANGE: Added featuremaps parameter
    P5 = KL.Conv2D(feature_maps, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(feature_maps, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(feature_maps, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(feature_maps, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(feature_maps, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    
    return [P2, P3, P4, P5, P6]


def pyramid_l1_graph(IP, T, feature_maps=128):
    L1P = []
    T = KL.AvgPool2D(pool_size=(int(T.shape[1]), int(T.shape[2])))(T)
    T = K.tile(T, [1, int(IP.shape[1]), int(IP.shape[2]), 1])
    L1P = K.abs(IP-T)
    # Currently no 1x1 conv after L1
#     L1P = KL.Conv2D(feature_maps, (1, 1), name='fpn_l1')(L1P)
    
    return L1P

def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Computer loss mean for 2 classes:
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

#modellib.mrcnn_class_loss_graph = mrcnn_class_loss_graph

class SiameseMaskRCNN(modellib.MaskRCNN):
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # CHANGE: add target input
        input_target = KL.Input(
            shape=config.TARGET_SHAPE.tolist(), name="input_target")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # CHANGE: Use weightshared FPN model for image and target
        # Create FPN Model
        resnet = build_resnet_model(self.config)
        fpn = build_fpn_model()
        # Create Image FP
        _, IC2, IC3, IC4, IC5 = resnet(input_image)
        IP2, IP3, IP4, IP5, IP6 = fpn([IC2, IC3, IC4, IC5])
        # Create Target FR
        _, TC2, TC3, TC4, TC5 = resnet(input_target)
        TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
        
        # CHANGE: add siamese distance copmputation
        # Combine FPs using L1 distance
        L1P2 = KL.Lambda(lambda x: pyramid_l1_graph(*x), name='pyramid_l1_graph_l1p2')([IP2, TP2])
        L1P3 = KL.Lambda(lambda x: pyramid_l1_graph(*x), name='pyramid_l1_graph_l1p3')([IP3, TP3])
        L1P4 = KL.Lambda(lambda x: pyramid_l1_graph(*x), name='pyramid_l1_graph_l1p4')([IP4, TP4])
        L1P5 = KL.Lambda(lambda x: pyramid_l1_graph(*x), name='pyramid_l1_graph_l1p5')([IP5, TP5])
        L1P6 = KL.Lambda(lambda x: pyramid_l1_graph(*x), name='pyramid_l1_graph_l1p6')([IP6, TP6])
    
        # CHANGE: combine original and siamese features
        P2 = KL.Lambda(lambda x: tf.concat([*x], axis=-1))([IP2, L1P2])
        P3 = KL.Lambda(lambda x: tf.concat([*x], axis=-1))([IP3, L1P3])
        P4 = KL.Lambda(lambda x: tf.concat([*x], axis=-1))([IP4, L1P4])
        P5 = KL.Lambda(lambda x: tf.concat([*x], axis=-1))([IP5, L1P5])
        P6 = KL.Lambda(lambda x: tf.concat([*x], axis=-1))([IP6, L1P6])

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        # CHANGE: Set number of filters to 256 [128 original + 128 L1]
        rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), 256)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = modellib.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: modellib.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: modellig.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                modellib.DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                modellib.fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = modellib.build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            # CHANGE: use custom class loss without using active_class_ids
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: modellib.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: modellib.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            # CHANGE: Added target to inputs
            inputs = [input_image, input_image_meta, input_target,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            # CHANGE: reduce number of classes to 2
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                modellib.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, num_classes=2,
                                     train_bn=config.TRAIN_BN)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
            # normalized coordinates
            detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            # CHANGE: reduce number of classes to 2
            mrcnn_mask = modellib.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              num_classes=2,
                                              train_bn=config.TRAIN_BN)

            # CHANGE: Added target to the input
            model = KM.Model([input_image, input_image_meta, input_target, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
    
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            modellib.log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                if verbose > 0:
                    print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
#             if trainable and verbose > 0:
#                 modellib.log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))
    
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gausssian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # region proposal network
            "rpn": r"(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "2+": r"(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        # CHANGE: Use siamese data generator
        train_generator = siamese_utils.siamese_data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = siamese_utils.siamese_data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        modellib.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        modellib.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
      
    
    def detect(self, targets, images, verbose=0):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            modellib.log("Processing {} images".format(len(images)))
            for image in images:
                modellib.log("image", image)
                # CHANGE: added target to logs
                modellib.log("target", np.stack(targets))

        # Mold inputs to format expected by the neural network
        # CHANGE: Removed moding of target -> detect expects molded target
        # TODO!
        molded_images, image_metas, windows = self.mold_inputs(images)
        # molded_targets, target_metas, target_windows = self.mold_inputs(targets)
        molded_targets = np.stack(targets)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
        # CHANGE: add size assertion for target
        target_shape = molded_targets[0].shape
        for g in molded_targets[1:]:
            assert g.shape == target_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            modellib.log("molded_images", molded_images)
#             modellib.log("image_metas", image_metas)
            # CHANGE: add targets to log
            modellib.log("molded_targets", molded_targets)
#             modellib.log("target_metas", target_metas)
            modellib.log("anchors", anchors)
        # Run object detection
        # CHANGE: Use siamese detection model
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, molded_targets, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results
    
    def evaluate_dataset(self, dataset, eval_type='detection', max_images=None, verbose=0):
    
        # Currently only batch size 1 ist supported
        assert self.config.BATCH_SIZE == 1, "Batch size must be 1"
        assert eval_type == 'detection', "Currently only detection is supported"
        assert self.mode == 'inference', "Model has to be in inference mode"

        # Permute image order
        image_ids = np.random.permutation(dataset.image_ids)
        # Limit number of images
        if max_images:
            image_ids = image_ids[:max_images]
            
        # Numerical stability factor:
        epsilon = 0.0001

        # Initialize lists to collect results
        # Get number of classes including BG
        nC = range(len(dataset.class_ids))
        # Initialize lists for averaged results
        coco_precisions = []
        precisions = []
        recalls = []
        false_positive_rates = []
        false_negative_rates = []
        jaccard_numbers = []
        # Initialize lists to collect results for every class
        number_of_instances = [[] for c in nC]
        number_of_predictions = [[] for c in nC]
        true_positives = [[] for c in nC]
        false_positives = [[] for c in nC]
        false_negatives = [[] for c in nC]

        # Iterate over images
        for image_id in image_ids:
            if verbose !=0:
                print(image_id)
            # Load GT data
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = modellib.load_image_gt(\
                                                                                         dataset, 
                                                                                         self.config, 
                                                                                         image_id, 
                                                                                         augmentation=False,
                                                                                         use_mini_mask=self.config.USE_MINI_MASK)

            # BOILERPLATE: Code duplicated in siamese_data_loader
            
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue
                
            # Use only positive class_ids
            categories = np.unique(gt_class_ids)
            _idx = categories > 0
            categories = categories[_idx]
            # Use only active classes
            active_categories = []
            for c in categories:
                if any(c == self.config.ACTIVE_CLASSES):
                    active_categories.append(c)
            
            # Skiop image if it contains no instance of any active class    
            if not np.any(np.array(active_categories) > 0):
                continue
                
            # END BOILERPLATE

            # Evaluate for every category individually
            for category in active_categories:

                # Draw random target
                target = siamese_utils.get_one_target(category, dataset, self.config)
                # Run siamese Mask R-CNN
                results = self.detect([target], [image], verbose=verbose)
                # Format detections
                r = results[0]

                # Select gt and detected boxes
                class_gt_boxes = gt_boxes[gt_class_ids == category]
                detected_boxes = r['rois']

                # Check IoUs to find correctly identified instances
                correct_ious = siamese_utils.find_correct_detections(class_gt_boxes, detected_boxes)
                # Select best matches
                # TODO: Replace with something smarter (what do we want?)
                best_matches_iou = siamese_utils.assign_detections(correct_ious)

    #             TP_at_threshold = []
    #             for tr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #                 TP_at_threshold.append(np.sum(np.array(siamese_utils.assign_detections(correct_ious, threshold=tr))))
    #             coco_precisions.append(TP_at_threshold)

                # Get copunts
                nI = class_gt_boxes.shape[0]
                nP  = detected_boxes.shape[0]
                TP = np.sum(np.array(best_matches_iou) > 0.5)
                FP = nP - TP
                FN = nI - TP
                number_of_instances[category].append(nI)
                number_of_predictions[category].append(nP)
                true_positives[category].append(TP)
                false_positives[category].append(FP)
                false_negatives[category].append(FN)
                # Calculate averaged statistics
                jaccard_index = TP / (TP + FP + FN)
                jaccard_numbers.append(jaccard_index)
                precision = TP / (nP + epsilon)
                precisions.append(precision)
                recall = TP / (nI + epsilon)
                recalls.append(recall)
                fpr = FP / (nP + epsilon)
                false_positive_rates.append(fpr)
                fnr = FN / (nI + epsilon)
                false_negative_rates.append(fnr)

                #print(true_positives, false_positives, false_negatives, number_of_instances)
                #print(jaccard_index, precision)

        # Print averaged Statistics   
        print('Averaged')
    #     print('AP: {:.2f}'.format(np.mean(coco_precisions)*100)) 
        print('AP50: {:.2f}'.format(np.mean(precisions)*100))
        print('AR50: {:.2f}'.format(np.mean(recalls)*100))
        print('JI50: {:.2f}'.format(np.mean(jaccard_numbers)*100))
        print('FPR50: {:.2f}'.format(np.mean(false_positive_rates)*100))
        print('FNR50: {:.2f}'.format(np.mean(false_negative_rates)*100))

        # Aggregate class-wise counts
        number_of_instances = np.array([np.sum(number_of_instances[c]) for c in nC])
        number_of_predictions = np.array([np.sum(number_of_predictions[c]) for c in nC])
        true_positives = np.array([np.sum(true_positives[c]) for c in nC])
        false_positives = np.array([np.sum(false_positives[c]) for c in nC])
        false_negatives = np.array([np.sum(false_negatives[c]) for c in nC])
        # Compute class-wise statistics
        jaccard_index = [(true_positives[c] + epsilon) /\
                         (true_positives[c] + false_positives[c] + false_negatives[c] + epsilon) for c in nC]
        precision = [(true_positives[c] + epsilon) / (number_of_predictions[c] + epsilon) for c in nC]
        recall = [(true_positives[c] + epsilon) / (number_of_instances[c] + epsilon) for c in nC]
        fpr = [false_positives[c] / (number_of_predictions[c] + epsilon) for c in nC]
        fnr = [(false_negatives[c] + epsilon) / (number_of_instances[c] + epsilon) for c in nC]
        # Remove unencountered classes
        idx = number_of_instances != 0
        jaccard_index = np.array(jaccard_index)[idx]
        precision = np.array(precision)[idx]
        recall = np.array(recall)[idx]
        fpr = np.array(fpr)[idx]
        fnr = np.array(fnr)[idx]
        # Print class-wise aggregated statistics
        print('')
        print('Aggregated per Class')
        print('AP50: {:.2f}'.format(np.mean(precision)*100))
        print('AR50: {:.2f}'.format(np.mean(recall)*100))
        print('JI50: {:.2f}'.format(np.mean(jaccard_index)*100))
        print('FPR50: {:.2f}'.format(np.mean(fpr)*100))
        print('FNR50: {:.2f}'.format(np.mean(fnr)*100))
        print('Encountered classes: {}'.format(np.sum(idx)))

        # Aggregate counts
        number_of_instances = np.sum(np.array([np.sum(number_of_instances[c]) for c in nC]))
        number_of_predictions = np.sum(np.array([np.sum(number_of_predictions[c]) for c in nC]))
        true_positives = np.sum(np.array([np.sum(true_positives[c]) for c in nC]))
        false_positives = np.sum(np.array([np.sum(false_positives[c]) for c in nC]))
        false_negatives = np.sum(np.array([np.sum(false_negatives[c]) for c in nC]))
        # Compoute aggregated statistics
        jaccard_index = (true_positives + epsilon) / (true_positives + false_positives + false_negatives + epsilon)
        precision = (true_positives + epsilon) / (number_of_predictions + epsilon)
        recall = (true_positives + epsilon) / (number_of_instances + epsilon)
        fpr = false_positives / (number_of_predictions + epsilon)
        fnr = (false_negatives + epsilon) / (number_of_instances + epsilon)
        #Print aggregated statistics
        print('')
        print('Aggregated')
        print('AP50: {:.2f}'.format(np.mean(precision)*100))
        print('AR50: {:.2f}'.format(np.mean(recall)*100))
        print('JI50: {:.2f}'.format(np.mean(jaccard_index)*100))
        print('FPR50: {:.2f}'.format(np.mean(fpr)*100))
        print('FNR50: {:.2f}'.format(np.mean(fnr)*100))