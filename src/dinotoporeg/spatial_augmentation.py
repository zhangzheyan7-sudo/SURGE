"""GDROS-style online spatial augmentation with explicit pixel-center GT mapping."""
import cv2
import numpy as np
import torch

def dense(matrix,h,w):
    y,x=torch.meshgrid(torch.arange(h,dtype=matrix.dtype),torch.arange(w,dtype=matrix.dtype),indexing='ij')
    return torch.stack((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]-x,
                        matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]-y))

def homogeneous(matrix):
    return torch.cat((matrix,matrix.new_tensor([[0.,0.,1.]])),0)

def resize_image(image,size):
    a=cv2.resize(image.permute(1,2,0).numpy(),(size[1],size[0]),interpolation=cv2.INTER_LINEAR)
    if a.ndim==2:a=a[...,None]
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).contiguous()

def crop_pair(opt,sar,flow,affine,top,left,size):
    ch,cw=size;h,w=opt.shape[-2:]
    if not (0<=top<=h-ch and 0<=left<=w-cw):raise ValueError('Crop outside input')
    t=affine.new_tensor([[1.,0.,-left],[0.,1.,-top],[0.,0.,1.]])
    matrix=(t@homogeneous(affine)@torch.linalg.inv(t))[:2]
    return tuple(x[...,top:top+ch,left:left+cw].contiguous() for x in (opt,sar,flow))+(matrix,)

def apply_spatial(opt,sar,flow,affine,size,hflip,vflip,top,left,crop_size):
    h,w=opt.shape[-2:];nh,nw=size
    if sar.shape[-2:]!=(h,w) or flow.shape[-2:]!=(h,w):raise ValueError('Image/flow size mismatch')
    if (nh,nw)!=(h,w):
        sx,sy=nw/w,nh/h
        # OpenCV half-pixel center convention. Account for actual rounded size.
        transform=affine.new_tensor([[sx,0.,(sx-1)/2],[0.,sy,(sy-1)/2],[0.,0.,1.]])
        updated=(transform@homogeneous(affine)@torch.linalg.inv(transform))[:2]
        # Preserve non-affine evidence from .flo, while extrapolating the known
        # affine component exactly at the boundary rather than clamping it.
        residual=flow-dense(affine,h,w)
        residual=resize_image(residual,size)*flow.new_tensor([sx,sy]).view(2,1,1)
        flow=dense(updated,nh,nw)+residual
        opt=resize_image(opt,size);sar=resize_image(sar,size);affine=updated
    if hflip:
        opt=opt.flip(-1);sar=sar.flip(-1);flow=flow.flip(-1)*flow.new_tensor([-1.,1.]).view(2,1,1)
        t=affine.new_tensor([[-1.,0.,nw-1],[0.,1.,0.],[0.,0.,1.]])
        affine=(t@homogeneous(affine)@t)[:2]
    if vflip:
        opt=opt.flip(-2);sar=sar.flip(-2);flow=flow.flip(-2)*flow.new_tensor([1.,-1.]).view(2,1,1)
        t=affine.new_tensor([[1.,0.,0.],[0.,-1.,nh-1],[0.,0.,1.]])
        affine=(t@homogeneous(affine)@t)[:2]
    return crop_pair(opt,sar,flow,affine,top,left,crop_size)

class GDROSSpatialAugmentor:
    def __init__(self,config):
        self.config=config
        self.crop=tuple(map(int,config.get('crop_size',[400,400])))
        if len(self.crop)!=2 or min(self.crop)<=0:raise ValueError('Invalid crop_size')
    def __call__(self,opt,sar,flow,affine):
        c=self.config;h,w=opt.shape[-2:];ch,cw=self.crop
        scale=2**np.random.uniform(c.get('min_scale',-.2),c.get('max_scale',.6))
        sx=sy=scale
        if np.random.rand()<c.get('stretch_probability',.8):
            sx*=2**np.random.uniform(-c.get('max_stretch',.2),c.get('max_stretch',.2))
            sy*=2**np.random.uniform(-c.get('max_stretch',.2),c.get('max_stretch',.2))
        minimum=max((ch+8)/h,(cw+8)/w)
        sx=max(sx,minimum);sy=max(sy,minimum)
        if np.random.rand()<c.get('spatial_probability',.8) or h<ch or w<cw:
            nh,nw=round(h*sy),round(w*sx)
        else:nh,nw=h,w
        hf=np.random.rand()<c.get('horizontal_flip_probability',.5)
        vf=np.random.rand()<c.get('vertical_flip_probability',.1)
        top=np.random.randint(0,nh-ch) if nh>ch else 0
        left=np.random.randint(0,nw-cw) if nw>cw else 0
        return apply_spatial(opt,sar,flow,affine,(nh,nw),hf,vf,top,left,self.crop)
