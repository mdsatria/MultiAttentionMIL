import numpy as np
import random
from tqdm.auto import tqdm


def check_bag(mnist, indices, class_label=1):
    labels = []
    for i in indices:
        _, lbl = mnist[i]
        labels.append(lbl)
    if class_label in labels:
        return True
    else:
        return False


def make_bags(mnist, num_of_bags, num_instances, label_bag):
    lst = [i for i in range(mnist.length)]

    images = []
    instance_label = []
    bag_label = []

    pbar_first = tqdm(range(num_of_bags // 2))
    for bag_id in pbar_first:
        found = False
        while found is False:
            random_indices = random.sample(lst, num_instances)
            check = check_bag(mnist, random_indices, label_bag)
            if check:
                found = True
                indices = random_indices
        img = [mnist[x][0] for x in indices]
        lbl = [mnist[x][1] for x in indices]
        images.append(img)
        instance_label.append(lbl)
        bag_label.append([label_bag])
        pbar_first.set_description("generating positive bags")

    temp_labels = np.array([mnist[x][1] for x in range(mnist.length)])
    lst = np.argwhere(temp_labels != label_bag).ravel().tolist()
    pbar_second = tqdm(range(num_of_bags // 2, num_of_bags))
    for bag_id in pbar_second:
        indices = random.sample(lst, num_instances)
        img = [mnist[x][0] for x in indices]
        lbl = [mnist[x][1] for x in indices]
        images.append(img)
        instance_label.append(lbl)
        bag_label.append([0])
        pbar_second.set_description("generating negative bags")

    return np.array(images), np.array(instance_label), np.array(bag_label)
