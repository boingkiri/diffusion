import jax
import flax
import flax.linen as nn

#TMP
TORCHVISION = [
    "vgg11_bn",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "densenet121",
    "densenet169",
    "densenet201",
    "inception_v3",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "shufflenet_v2_x0_5",
    "mobilenet_v2",
    "wide_resnet50_2",
    "mnasnet0_5",
    "mnasnet1_0",
    "ghostnet_100",
    "cspresnet50",
    "fbnetc_100",
    "spnasnet_100",
    "resnet50d",
    "resnet26",
    "resnet26d",
    "seresnet50",
    "resnetblur50",
    "resnetrs50",
    "tf_mixnet_s",
    "tf_mixnet_m",
    "tf_mixnet_l",
    "ese_vovnet19b_dw",
    "ese_vovnet39b",
    "res2next50",
    "gernet_s",
    "gernet_m",
    "repvgg_a2",
    "repvgg_b0",
    "repvgg_b1",
    "repvgg_b1g4",
    "revnet",
    "dm_nfnet_f1",
    "nfnet_l0",
]
REGNETS = [
    "regnetx_002",
    "regnetx_004",
    "regnetx_006",
    "regnetx_008",
    "regnetx_016",
    "regnetx_032",
    "regnetx_040",
    "regnetx_064",
    "regnety_002",
    "regnety_004",
    "regnety_006",
    "regnety_008",
    "regnety_016",
    "regnety_032",
    "regnety_040",
    "regnety_064",
]
EFFNETS_IMAGENET = [
    'tf_efficientnet_b0',
    'tf_efficientnet_b1',
    'tf_efficientnet_b2',
    'tf_efficientnet_b3',
    'tf_efficientnet_b4',
    'tf_efficientnet_b0_ns',
]
EFFNETS_INCEPTION = [
    'tf_efficientnet_lite0',
    'tf_efficientnet_lite1',
    'tf_efficientnet_lite2',
    'tf_efficientnet_lite3',
    'tf_efficientnet_lite4',
    'tf_efficientnetv2_b0',
    'tf_efficientnetv2_b1',
    'tf_efficientnetv2_b2',
    'tf_efficientnetv2_b3',
    'efficientnet_b1',
    'efficientnet_b1_pruned',
    'efficientnet_b2_pruned',
    'efficientnet_b3_pruned',
]
EFFNETS = EFFNETS_IMAGENET + EFFNETS_INCEPTION
VITS_IMAGENET = [
    'deit_tiny_distilled_patch16_224',
    'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224',
]
VITS_INCEPTION = [
    'vit_base_patch16_224'
]
VITS = VITS_IMAGENET + VITS_INCEPTION
CLIP = [
    'resnet50_clip'
]
ALL_MODELS = TORCHVISION + REGNETS + EFFNETS + VITS + CLIP

# Group according to input normalization
NORMALIZED_IMAGENET = TORCHVISION + REGNETS + EFFNETS_IMAGENET + VITS_IMAGENET
NORMALIZED_INCEPTION = EFFNETS_INCEPTION + VITS_INCEPTION
NORMALIZED_CLIP = CLIP

def _make_projector(im_res, backbone, cout, proj_type, expand=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    pretrained = _make_pretrained(backbone)

    # Following Projected GAN
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]

    if proj_type == 0: return pretrained, None

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)

    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


def get_backbone_normstats(backbone):
    if backbone in NORMALIZED_INCEPTION:
        return {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        }

    elif backbone in NORMALIZED_IMAGENET:
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }

    elif backbone in NORMALIZED_CLIP:
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        }

    else:
        raise NotImplementedError


class F_RandomProj(nn.Module):
    def __init__(
        self,
        backbone="tf_efficientnet_lite3",
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.cout = cout
        self.expand = expand
        self.normstats = get_backbone_normstats(backbone)

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, backbone=self.backbone, cout=self.cout,
                                                        proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        # predict feature maps
        if self.backbone in VITS:
            out0, out1, out2, out3 = forward_vit(self.pretrained, x)
        else:
            out0 = self.pretrained.layer0(x)
            out1 = self.pretrained.layer1(out0)
            out2 = self.pretrained.layer2(out1)
            out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out

# class ProjectedDiscriminator(nn.Module):

