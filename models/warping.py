'''
This file is an implementation of the following paper:

	Deep High-Resolution Representation Learning for Visual Recognition, 2020
	https://arxiv.org/pdf/1908.07919.pdf
    
	Learning Temporal Pose Estimation from Sparsely-Labeled Videos, 2019
	https://arxiv.org/pdf/1906.04016.pdf

	By Shuchen Du

'''
import torch

from torch import nn
from .deform_conv import DeformConv


BN_MOMENTUM = 0.1


class Conv(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, relued=True):
		super().__init__()
		padding = (kernel_size - 1) // 2
		self.conv_bn = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
				nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM))
		self.relu = nn.ReLU()
		self.relued = relued

	def forward(self, x):
		x = self.conv_bn(x)
		if self.relued:
			x = self.relu(x)
		return x


class BasicBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = nn.Sequential(
				Conv(in_ch, out_ch),
				Conv(out_ch, out_ch, relued=False))
		self.relu = nn.ReLU()
		self.downsampling = None
		if in_ch != out_ch:
			self.downsampling = Conv(in_ch, out_ch, 1, relued=False)

	def forward(self, x):
		identity = x
		x = self.conv(x)
		if self.downsampling:
			identity = self.downsampling(identity)
		x += identity
		return self.relu(x)


class Warping(nn.Module):
	def __init__(self, out_ch=26):
		super().__init__()

		# offset optimization
		inner_ch = 128
		self.offset_feats = self._compute_chain_of_basic_blocks(out_ch, inner_ch, 20)

		k = 3

		#### warping
		self.offsets = nn.ModuleList()
		self.deform_convs = nn.ModuleList()
		dilations = [3, 6, 12, 18, 24]
		for i in range(5):
			self.offsets.append(self._dilated_conv(inner_ch, k, k, dilations[i], out_ch))
			self.deform_convs.append(self._deform_conv(out_ch, k, k, dilations[i], out_ch))

		# init layers
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, std=0.001)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _dilated_conv(self,  nc, kh, kw, dd, dg):
		conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd), padding=(1*dd, 1*dd), bias=False)
		return conv

	def _deform_conv(self, nc, kh, kw, dd, dg):
		conv_offset2d = DeformConv(nc, nc, (kh, kw), stride=1, padding=int(kh/2)*dd, dilation=dd, deformable_groups=dg)
		return conv_offset2d

	def _compute_chain_of_basic_blocks(self, in_ch, out_ch, num_blocks):
		layers = [BasicBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(num_blocks)]
		return nn.Sequential(*layers)

	def forward(self, ref_x, sup_x):
		# offset
		diff_x = ref_x - sup_x

		# offset optimization
		off_feats = self.offset_feats(diff_x)

		# warping
		warped_x_list = []
		for offset, deform_conv in zip(self.offsets, self.deform_convs):
			warped_x_list.append(deform_conv(sup_x, offset(off_feats)))

		return torch.mean(torch.stack(warped_x_list), dim=0)

