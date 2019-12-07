   # validate process
    for i, data in enumerate(data_loader_test):
        inputs, ground_truth = data
        inputs = torch.transpose(inputs.cuda(), 3, 1)
        outputs = model(inputs.float())
        outputs = outputs.view(outputs.shape[1:])
        outputs = torch.transpose(outputs, 2, 1)

        # calculate iou
        outputs[outputs > 0] = 1
        outputs[outputs < 0] = 0
        temp = outputs + ground_truth.cuda()
        intersection = torch.zeros_like(outputs)
        union = torch.zeros_like(outputs)
        # print(outputs.shape)
        intersection[temp >= 1] = 1
        union[temp == 2] = 1
        iou = torch.sum(union)/ torch.sum(intersection)
        print("epoch:{}, iou:{}".format(epoch,iou))

        # save model's parameter and output image at each 'num_epochs_save' step
        if (epoch + 1) % num_epochs_save == 0:
            pkl_file_name = "epoch_{}_iou_{}.pkl".format(epoch+1,iou)
            pkl_file_path = os.path.join(pth_file_dir,pkl_file_name)
            torch.save(model.state_dict(), pkl_file_path)
            # torch.save(model.state_dict(), ".\\model_pth\\epoch_{}_iou_{}.pkl".format(epoch,iou))

            # save output image
            output_image_name = "{}_{}.png".format(epoch+1, i)
            result_file_path = os.path.join(val_image_dir,output_image_name)
            outputs = outputs.cpu().detach().numpy()
            plt.imsave(result_file_path,outputs.reshape(outputs.shape[1:]))


        if iou >= accuracy_exceed_precent :
            accuracy_exceed_precent_flag = True
    
    # stop train and save model's parameter
    if (accuracy_exceed_precent_flag == True and is_allow_end_train == True) or epoch+1 == num_epochs :
        pkl_file_name = "final_epoch_{}_iou_{}.pkl".format(epoch,iou)
        pkl_file_path = os.path.join(pth_file_dir,pkl_file_name)
        torch.save(model.state_dict(), pkl_file_path)

        output_image_name = "final_epoch_{}_iou_{}.png".format(epoch, i)
        result_file_path = os.path.join(val_image_dir,output_image_name)
        if epoch+1 == 100:
            outputs = outputs.detach().numpy()
        plt.imsave(result_file_path,outputs.reshape(outputs.shape[1:]))
        break