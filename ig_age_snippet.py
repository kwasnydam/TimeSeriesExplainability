if args.explain_prediction:
        pytorch_embedder = embedder.to_pytorch()
        pytorch_embedder.eval()

        dataloader_iterator = iter(data_layer_test.data_iterator)
        cuda0 = torch.device('cuda:0') 
        test_input_waves, test_input_length, class_label = next(dataloader_iterator)

        test_input_waves = test_input_waves.to(cuda0)
        class_label = class_label.to(cuda0)
        
        test_input_tensor, _ = data_preprocessor.forward(test_input_waves, test_input_length)
        with torch.no_grad():
            prediction = pytorch_embedder.forward(test_input_tensor)
            logger.info(f"network prediction: {prediction.detach().cpu().numpy()}")
            
            real_class = post_process_class_labels(class_label, labels)
            predicted_class = post_process_predictions(prediction.detach(), labels)
            logger.info(f"predicted: {predicted_class}, real class: {real_class}")
        
        ig = IntegratedGradients(pytorch_embedder)
        
        test_input_tensor.requires_grad_()        
        attr, delta = ig.attribute(test_input_tensor, target=class_label, return_convergence_delta=True)
        attr = attr.cpu().detach().numpy()

        # plot the results
        fig, axs = plt.subplots(3, 1, sharex=False, sharey=False)
        dpi = 200
        logger.info(f"shape of the axs: {axs.shape}")
        axs[0].plot([i/sample_rate for i in range(test_input_waves[0].shape[0])], 
        test_input_waves[0].cpu().detach().numpy())

        axs[1].imshow(test_input_tensor.cpu().detach().numpy()[0], cmap=cm.coolwarm, origin='lower')
        axs[1].set_title('Network input features')
        axs[1].set_ylabel('Value')
        logger.info(f"shape of the attribute importance matrix is: {attr.shape}")
        
        axs[2].imshow(attr[0], cmap=cm.coolwarm, origin='lower')
        axs[2].set_title('Integrated Gradients attributes')
        axs[2].set_xlabel('Frame Number (1 frame - 25 ms)')
        axs[2].set_ylabel('Value')
        
        plt.savefig(os.path.join(args.save_directory, 'explain_matrix_1.png'), dpi=dpi)
