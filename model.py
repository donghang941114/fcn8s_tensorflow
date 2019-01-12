model = FCN8s(model_load_dir=None,
              tags=None,
              vgg16_dir='../VGG-16_mod2FCN_ImageNet-Classification',
              num_classes=num_classes,
              variables_load_dir=None)

epochs = 6  # TODO: Set the number of epochs to train for.


# TODO: Define a learning rate schedule function to be passed to the `train()` method.
def learning_rate_schedule(step):
    if step <= 10000:
        return 0.0001
    elif 10000 < step <= 20000:
        return 0.00001
    elif 20000 < step <= 40000:
        return 0.000003
    else:
        return 0.000001


model.train(train_generator=train_generator,
            epochs=epochs,
            steps_per_epoch=ceil(num_train_images / batch_size),
            learning_rate_schedule=learning_rate_schedule,
            keep_prob=0.5,
            l2_regularization=0.0,
            eval_dataset='val',
            eval_frequency=2,
            val_generator=val_generator,
            val_steps=ceil(num_val_images / batch_size),
            metrics={'loss', 'mean_iou', 'accuracy'},
            save_during_training=True,
            save_dir='cityscapes_model',
            save_best_only=True,
            save_tags=['default'],
            save_name='(batch-size-4)',
            save_frequency=2,
            saver='saved_model',
            monitor='loss',
            record_summaries=True,
            summaries_frequency=10,
            summaries_dir='tensorboard_log/cityscapes',
            summaries_name='configuration_01',
            training_loss_display_averaging=3)