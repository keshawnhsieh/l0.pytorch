from models import L0Net, LeNet, VGG, L0VGG

### Global Parameters
g_mean = -0.5
g_coef = 0.002
g_epochs = 200
g_batch_size = 128
g_temp = 0.33

cifar10_network = "VGG16"

### PT file <-> Structure Mapping, Used by Pruning and Finetune, MaskCheck
def pt2stru(pt):
    if "lenet" in pt:
        return LeNet()
    elif "l0net" in pt:
        return L0Net(mean=1)
    elif "VGG" in pt:
        if "l0" in pt:
            return L0VGG(cifar10_network, loc=g_mean, temp=g_temp)
        else:
            return VGG(cifar10_network)

### PT file <-> Dataset Name, Used by Pruning and Finetune
def pt2data(pt):
    if "lenet" in pt or "l0net" in pt:
        return "mnist"
    elif "VGG" in pt:
        return "cifar10"

### Structure
def get_network(name, baseline=True, **kwargs):
    mean = kwargs.get("mean")
    temp = kwargs.get("temp")
    if name == "mnist":
        if baseline: return LeNet()
        else:
            try:
                return L0Net(mean=mean, temp=temp)
            except KeyError:
                print("No Key Named 'mean'")
    elif name == "cifar10":
        if baseline: return VGG(cifar10_network)
        else:
            return L0VGG(cifar10_network, loc=mean, temp=temp)

### Training Checkpoint
def training_checkpoint(name, baseline=True):
    if name == "mnist":
        if baseline: return "pt/lenet.pt"
        else: return "pt/l0net.pt"
    elif name == "cifar10":
        if baseline: return "pt/{}.pt".format(cifar10_network)
        else: return "pt/l0{}.pt".format(cifar10_network)

### Pruning Checkpoint
def pruning_checkpoint(pt, rate):
    name = pt.split(".")[0]
    return "{name}_p{rate}.pt".format(
        name=name,
        rate=rate
    )

### Finetune Checkpoint
def finetune_checkpoint(pt):
    name = pt.split(".")[0]
    return "{name}_finetune.pt".format(
        name=name
    )

### Logger Dir
def training_logger_dir(name, baseline, mean, temp, coef):
    return "runs/{name}_{type}_coef<{coef}>_mean<{mean}>_temp<{temp}>".format(
        name=name,
        type="bs" if baseline else "l0",
        coef=coef,
        mean=mean,
        temp=temp
    )

def pruning_logger_dir(pt, rate):
    if "lenet" in pt or "l0net" in pt:
        return "runs/mnist_{type}_coef<{coef}>_mean<{mean}>_temp<{temp}>_rate<{rate}>".format(
            type="bs" if "l0" not in pt else "l0",
            coef=g_coef,
            mean=g_mean,
            temp=g_temp,
            rate=rate
        )
    elif "VGG" in pt:
        return "runs/cifar10_{type}_coef<{coef}>_mean<{mean}>_temp<{temp}>_rate<{rate}>".format(
            type="bs" if "l0" not in pt else "l0",
            coef=g_coef,
            mean=g_mean,
            temp=g_temp,
            rate=rate
        )

def finetune_logger_dir(pt):
    rate = pt.split("_p")[-1][:2]
    if "lenet" in pt or "l0net" in pt:
        return "runs/mnist_{type}_coef<{coef}>_mean<{mean}>_temp_<{temp}>_rate<{rate}>_finetune".format(
            type="bs" if "l0" not in pt else "l0",
            coef=g_coef,
            mean=g_mean,
            temp=g_temp,
            rate=rate
        )
    elif "VGG" in pt:
        return "runs/cifar10_{type}_coef<{coef}>_mean<{mean}>_temp<{temp}>_rate<{rate}>_finetune".format(
            type="bs" if "l0" not in pt else "l0",
            coef=g_coef,
            mean=g_mean,
            temp=g_temp,
            rate=rate
        )