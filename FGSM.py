"""
The implementation of FGSM attack (Ian Goodfellow et al. Explaining and harnessing adversarialexamples. InICLR, 2015.)
"""

def fgsm_attack(image, epsilon, data_grad, upper_bound, lower_bound):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    for c in range(3):
        perturbed_image[:, c, :, :][perturbed_image[:, c, :, :] < lower_bound[c, 0, 0]] = lower_bound[c, 0, 0]
        perturbed_image[:, c, :, :][perturbed_image[:, c, :, :] > upper_bound[c, 0, 0]] = upper_bound[c, 0, 0]
    # Return the perturbed image
    return perturbed_image
