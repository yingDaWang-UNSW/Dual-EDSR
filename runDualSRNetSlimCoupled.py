'''
Double 2D super resolution method
'''

#TODO: write up testing section if train if test. enable substacking!
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import dualSRNetArgs
import tifffile
# Helper libraries
from sys import stdout
import numpy as np
import os
from glob import glob
import time
import datetime
import pdb
import imageio
from matplotlib import pyplot as plt


AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

args=dualSRNetArgs.args() # args is global

gpuList=args.gpuIDs
args.numGPUs = len(gpuList.split(','))
if args.numGPUs<=4:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuList

if args.mixedPrecision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
else:
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# detect hardware
if len(args.gpuIDs.split(','))<=1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# define the network
with strategy.scope():
    # define functions used
    def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


    def apply_blur(img,kernel_size, sigma, n_channel):
        blur = _gaussian_kernel(kernel_size, sigma, n_channel, img.dtype)
        img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
        return img

    def random_croptf2(image, height, width):
        cropped_image = tf.image.random_crop(image, size=[image.shape[0], np.min((height,image.shape[1])), np.min((width,image.shape[2])), image.shape[3]])
        return cropped_image
        
    def random_croptf23D(image, height, width, depth):
        x=int(tf.floor(np.random.rand()*(image.shape[0]-height)))
        y=int(tf.floor(np.random.rand()*(image.shape[1]-width)))
        z=int(tf.floor(np.random.rand()*(image.shape[2]-depth)))
        cropped_image=tf.expand_dims(image[x:x+height,y:y+width,z:z+depth,:],0)
        #cropped_image = tf.image.random_crop(image, size=[image.shape[0], np.min((height,image.shape[1])), np.min((width,image.shape[2])), image.shape[3]])
        return cropped_image
    
    def createTrainingCubes2(args,HR,LRxy,batchsize,cropsize,scale):
        # read an HR block and extract the LRxy,LRyz, and LRxz blocks of size itersperepoch*batch,x,y,1
        # permute the block so the lrbc dimension is in the batch dimension
        batchLR = np.zeros([batchsize*args.itersPerEpoch,cropsize,cropsize,1],'float32')
        batchHR = np.zeros([batchsize*args.itersPerEpoch*scale,cropsize*scale,cropsize*scale,1],'float32')
        n=0
        n2=0
        for i in range(args.itersPerEpoch):
            # cycle between xy,yz, and xz for extra data - first version was fucked because batch is explicitly the bc dim but it wasnt in this implementation 
            if np.mod(i,3)==0:
                x=int(np.floor(np.random.rand()*(LRxy.shape[0]-batchsize)))
                y=int(np.floor(np.random.rand()*(LRxy.shape[1]-cropsize)))
                z=int(np.floor(np.random.rand()*(LRxy.shape[2]-cropsize)))
                
                block=np.expand_dims(LRxy[x:x+batchsize,y:y+cropsize,z:z+cropsize],3)
                blockHR=np.expand_dims(HR[x*scale:x*scale+batchsize*scale,y*scale:y*scale+cropsize*scale,z*scale:z*scale+cropsize*scale],3)

            elif np.mod(i,3)==1:
                x=int(np.floor(np.random.rand()*(LRxy.shape[0]-cropsize)))
                y=int(np.floor(np.random.rand()*(LRxy.shape[1]-cropsize)))
                z=int(np.floor(np.random.rand()*(LRxy.shape[2]-batchsize)))
                
                block=np.expand_dims(LRxy[x:x+cropsize,y:y+cropsize,z:z+batchsize],3)
                blockHR=np.expand_dims(HR[x*scale:x*scale+cropsize*scale,y*scale:y*scale+cropsize*scale,z*scale:z*scale+batchsize*scale],3)
                block=np.transpose(block,[2,0,1,3])
                blockHR=np.transpose(blockHR,[2,0,1,3])
                

            elif np.mod(i,3)==2:
                x=int(np.floor(np.random.rand()*(LRxy.shape[0]-cropsize)))
                y=int(np.floor(np.random.rand()*(LRxy.shape[1]-batchsize)))
                z=int(np.floor(np.random.rand()*(LRxy.shape[2]-cropsize)))
                
                block=np.expand_dims(LRxy[x:x+cropsize,y:y+batchsize,z:z+cropsize],3)
                blockHR=np.expand_dims(HR[x*scale:x*scale+cropsize*scale,y*scale:y*scale+batchsize*scale,z*scale:z*scale+cropsize*scale],3)
                
                block=np.transpose(block,[1,0,2,3])
                blockHR=np.transpose(blockHR,[1,0,2,3])
                
            batchLR[n:n+batchsize]=block/127.5-1
            batchHR[n2:n2+batchsize*scale]=blockHR/127.5-1
            #batchLR[n:n+batchsize]=block*2-1
            #batchHR[n:n+batchsize]=blockHR*2-1
            n=n+batchsize
            n2=n2+batchsize*scale
            
            stdout.write("\rHR Cube: %d of %d" % (i+1, args.itersPerEpoch))
            stdout.flush()
        stdout.write("\n")
        return batchHR,batchLR
        
    def augmentData(image):
      #image = tf.image.random_contrast(image, 0.8, 1.2)
      #image = tf.image.random_brightness(image, 0.4)
      #image = image + tf.random.normal(image.shape,0,0.05)
      #sigma = np.random.rand()*2
      #image = apply_blur(image,2*np.ceil(2*sigma)+1, sigma, 1)
      # inject random contrast and brightness adjustments
      contFactor = (np.random.rand()*2-1)*0.2+1
      brightFactor = (np.random.rand()*2-1)*0.2+1
      
      image = image*brightFactor
      
      image = (image-tf.math.reduce_mean(image))*contFactor + tf.math.reduce_mean(image)
      image = tf.clip_by_value(image,-1,1)
      return image

        
    # define architecture
    
    def conv(ndims, *args, **kwargs):
        if ndims==2:
            return  tf.keras.layers.Conv2D(*args, **kwargs)
        elif ndims==3:
            return  tf.keras.layers.Conv3D(*args, **kwargs)
    
    class InstanceNormalization(tf.keras.layers.Layer):
      """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

      def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

      def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

      def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
        
    class InstanceNormalization3D(tf.keras.layers.Layer):
      """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

      def __init__(self, epsilon=1e-5):
        super(InstanceNormalization3D, self).__init__()
        self.epsilon = epsilon

      def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

      def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
        
    def instanceNorm(x,ndims):
        if ndims==2:
            x = InstanceNormalization()(x)
        elif ndims==3:
            x = InstanceNormalization3D()(x)
        return x
        
    def res_block_EDSR(x_in, filters, kernel, norm_type='instancenorm', apply_norm=False, ndims=2):
        x = conv(ndims, filters, kernel, padding='same')(x_in)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = instanceNorm(x,ndims)
        x = conv(ndims, filters, kernel, padding='same')(x)
        x = tf.keras.layers.Add()([x_in, x])
        return x
        
    def upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
        def upsample_edsr(x, factor, ndims, **kwargs):
            #zx = conv(ndims, num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
            x = tf.keras.layers.Activation('relu')(x)
            if apply_norm:
                if norm_type.lower() == 'batchnorm':
                    x = tf.keras.layers.BatchNormalization()(x)
                elif norm_type.lower() == 'instancenorm':
                    x = instanceNorm(x,ndims)
            if ndims==2:
                x = tf.keras.layers.UpSampling2D(size=factor)(x)
                #x = SubpixelConv2D(factor)(x)
                return x
            elif ndims==3:
                x = tf.keras.layers.UpSampling3D(size=factor)(x)
                return x
                
        if scale == 2:
            x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
        elif scale == 3:
            x = upsample_edsr(x, 3, ndims=ndims, name='conv2d_1_scale_3_up'+nameIn)
        elif scale == 4:
            x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
            x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
        elif scale == 8:
            x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
            x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
            x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_3_scale_2_up'+nameIn)
        return x
        
        
    def upsampleEDSR1D(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
        def upsample_edsr(x, factor, ndims, **kwargs):
            #zx = conv(ndims, num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
            x = tf.keras.layers.Activation('relu')(x)
            if apply_norm:
                if norm_type.lower() == 'batchnorm':
                    x = tf.keras.layers.BatchNormalization()(x)
                elif norm_type.lower() == 'instancenorm':
                    x = instanceNorm(x,ndims)
            if ndims==2:
                x = tf.keras.layers.UpSampling2D(size=factor)(x)
                #x = SubpixelConv2D(factor)(x)
                return x
            elif ndims==3:
                x = tf.keras.layers.UpSampling3D(size=factor)(x)
                return x
                
        if scale == 2:
            x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
        elif scale == 3:
            x = upsample_edsr(x, (3,1), ndims=ndims, name='conv2d_1_scale_3_up'+nameIn)
        elif scale == 4:
            x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
            x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
        elif scale == 8:
            x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
            x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
            x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_3_scale_2_up'+nameIn)
        return x
        
        
    def SubpixelConv2D(scale, **kwargs):
        return  tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)

    def edsr(scale, num_filters=64, num_res_blocks=8, ndims=2):
        if ndims==2:
            x_in = tf.keras.layers.Input(shape=(None, None, 1))
        elif ndims==3:
            x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
        x = x_in
        x = b = conv(ndims, num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False, ndims=ndims)
        b = conv(ndims, num_filters, 3, padding='same')(b)
        x = tf.keras.layers.Add()([x, b])

        x = upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=ndims)
        
#        x = conv(ndims, num_filters, 3, padding='same')(x)
#        x = tf.keras.layers.Activation('relu')(x)
        
        x = conv(ndims, 1, 3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
        
        return tf.keras.models.Model(x_in, x, name="EDSR")
        
    def edsr1D(scale, num_filters=64, num_res_blocks=8, ndims=2):
        if ndims==2:
            x_in = tf.keras.layers.Input(shape=(None, None, 1))
        elif ndims==3:
            x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
        x = x_in
        x = b = conv(ndims, num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False, ndims=ndims)
        b = conv(ndims, num_filters, 3, padding='same')(b)
        x = tf.keras.layers.Add()([x, b])

        x = upsampleEDSR1D(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=ndims)
        
#        x = conv(ndims, num_filters, 3, padding='same')(x)
#        x = tf.keras.layers.Activation('relu')(x)
        
        x = conv(ndims, 1, 3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
        
        return tf.keras.models.Model(x_in, x, name="EDSR")
      
    def disc_block(x_in, filters, ndims):
        x = conv(ndims, filters, 3, 1, padding='same')(x_in)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = conv(ndims, filters, 3, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x
        
    def DiscriminatorSRGAN(args):
        #xIn = tf.keras.layers.Input(shape=[args.fine_size, args.fine_size, args.output_nc], name='Disc_Inputs')

        xIn = tf.keras.layers.Input(shape=[args.disc_size, args.disc_size, 1], name='Disc_Inputs')

        # shallow layers
        x = conv(2, args.ndsrf, 3, 1, padding='same')(xIn)
        x = tf.keras.layers.LeakyReLU()(x)
        x = conv(2, args.ndsrf, 3, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        numDiscBlocks=3
        for i in range(numDiscBlocks):
            x = disc_block(x, args.ndsrf*(2**(i+1)), 2)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        xOut = tf.keras.layers.Dense(1, dtype='float32')(x)
        '''
        h = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='dInitConv'))
        h = lrelu(batchnormSR(conv2d(h, options.df_dim, ks=3, s=s, name='dUpConv')))
        for i in range(numDiscBlocks):
            expon=2**(i+1)
            h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=1, name=f'dBlock{i+1}Conv')))
            h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=2, name=f'dBlock{i+1}UpConv')))
        h = conv2d(h, 1, ks=3, s=1, name='d_h3_pred')
        #h = lrelu(denselayer(slim.flatten(h), 1024, name="dFC1"))
        #h = denselayer(h, 1, name="dFCout")
        return h
        
        '''  
        return tf.keras.Model(inputs=[xIn], outputs=xOut, name="DiscrimSR")
        
    def DiscriminatorSRGAN3D(args):
        #xIn = tf.keras.layers.Input(shape=[args.fine_size, args.fine_size, args.output_nc], name='Disc_Inputs')

        xIn = tf.keras.layers.Input(shape=[args.disc_size//2, args.disc_size//2, args.disc_size//2, 1], name='Disc_Inputs')

        # shallow layers
        x = conv(3, args.ndsrf, 3, 1, padding='same')(xIn)
        x = tf.keras.layers.LeakyReLU()(x)
        x = conv(3, args.ndsrf, 3, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        numDiscBlocks=3
        for i in range(numDiscBlocks):
            x = disc_block(x, args.ndsrf*(2**(i+1)), 3)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        xOut = tf.keras.layers.Dense(1, dtype='float32')(x)

        return tf.keras.Model(inputs=[xIn], outputs=xOut, name="DiscrimSR3D")
        


    # standard losses
    def meanAbsoluteError(labels, predictions):
        per_example_loss = tf.reduce_mean(tf.abs(labels-predictions), axis = [1,2,3]) # could not be bothered to softcode this...
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])
        
    def sigmoidCrossEntropy(labels, logits):
        per_example_sxe = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis = 1)
        return tf.nn.compute_average_loss(per_example_sxe, global_batch_size=args.batch_size)
        
    def scganLoss(disc_real_output, disc_generated_output):
        real_loss = sigmoidCrossEntropy(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
        return 0.5*total_disc_loss
        
    def advScganLoss(disc_generated_output):
        adversarial_loss = sigmoidCrossEntropy(tf.ones_like(disc_generated_output), disc_generated_output)
        return adversarial_loss
        
    def createSRGenerator(args):

        generator = edsr(scale=args.scale, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks, ndims=2)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(lr=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator, loss_scale='dynamic')
        return generator, optimizerGenerator            
        
    def createSRCGenerator(args):

        generator = edsr1D(scale=args.scale, num_filters=args.ngsrf//2, num_res_blocks=args.numResBlocks//2, ndims=2)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(lr=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator, loss_scale='dynamic')
        return generator, optimizerGenerator            

    def createSRDiscriminator(args):
        if args.ganFlag:
            discriminator = DiscriminatorSRGAN(args)
            optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)    
        else:
            a = tf.keras.layers.Input(shape=(1,))
            b = a
            discriminator = tf.keras.models.Model(inputs=a, outputs=b)
            optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)        
        optimizerDiscriminator = mixed_precision.LossScaleOptimizer(optimizerDiscriminator, loss_scale='dynamic')     
        discriminator.summary(200)
        return discriminator, optimizerDiscriminator
        
    def createSRDiscriminator3D(args):
        if args.ganFlag:
            discriminator = DiscriminatorSRGAN3D(args)
            optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)    
        else:
            a = tf.keras.layers.Input(shape=(1,))
            b = a
            discriminator = tf.keras.models.Model(inputs=a, outputs=b)
            optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)        
        optimizerDiscriminator = mixed_precision.LossScaleOptimizer(optimizerDiscriminator, loss_scale='dynamic')     
        discriminator.summary(200)
        return discriminator, optimizerDiscriminator

    # define the actions taken per iteration (calc grads and make an optim step)
    def train_step(HRBatch,BCBatch):
        Cxyz, Bxy =  HRBatch, BCBatch # make sure the dims are correct
        if args.augFlag:
            Bxy = augmentData(Bxy)
        # train
        with tf.GradientTape(persistent=True) as tape:
            # run a cycle on the cycleGAN
            totalGsrXYLoss = 0
            totalGsrYZLoss = 0
            
            advsrXYLoss = 0
            dsrXYLoss = 0

            
            advsrYZLoss = 0
            dsrYZLoss = 0

            
            Cxyd=tf.image.resize(tf.squeeze(Cxyz),[Cxyz.shape[0]//args.scale,Cxyz.shape[2]],method='bicubic')
            Cxyd=tf.expand_dims(Cxyd,3)
            SRxy = generatorSR(Bxy, training=True)
            totalGsrXYLoss = meanAbsoluteError(Cxyd, SRxy)
            
            if args.ganFlag:
                disc_C = discriminatorSR(random_croptf2(Cxyd, args.disc_size, args.disc_size), training=True)
                disc_BASR = discriminatorSR(random_croptf2(SRxy, args.disc_size, args.disc_size), training=True)
                
                advsrXYLoss = advsrXYLoss + advScganLoss(disc_BASR)
                dsrXYLoss = dsrXYLoss + scganLoss(disc_C, disc_BASR)
                
                totalGsrXYLoss = totalGsrXYLoss + args.srAdv_lambda*advsrXYLoss
                
                
            # set bit depth to 8 for SRxy
            SRxy=(SRxy+1)*127.5
            SRxy=tf.math.round(SRxy)
            SRxy=SRxy/127.5 - 1
            # transpose the volume
            SRxy = tf.transpose(SRxy,perm=[1,0,2,3])
            Cxyz = tf.transpose(Cxyz,perm=[1,0,2,3])
            
            # resize the slices
            # SRxyd=tf.image.resize(SRxy,[SRxy.shape[1],SRxy.shape[2]//args.scale],method='bicubic')
            
            SRxyz = generatorSRC(SRxy, training=True)
            totalGsrYZLoss = meanAbsoluteError(Cxyz, SRxyz)
         
            if args.ganFlag:
                disc_CC = discriminatorSRC(random_croptf23D(Cxyz, args.disc_size//2, args.disc_size//2, args.disc_size//2), training=True)
                disc_BASRC = discriminatorSRC(random_croptf23D(SRxyz, args.disc_size//2, args.disc_size//2, args.disc_size//2), training=True)
                
                advsrYZLoss = advsrYZLoss + advScganLoss(disc_BASRC)
                dsrYZLoss = dsrYZLoss + scganLoss(disc_CC, disc_BASRC)
                
                totalGsrYZLoss = totalGsrYZLoss + args.srAdv_lambda*advsrYZLoss
            
            totalGsrXYZLoss = totalGsrYZLoss + totalGsrXYLoss
                        
            totalGsrLossScal = optimizerGeneratorSR.get_scaled_loss(totalGsrXYZLoss)
            totalGsrcLossScal = optimizerGeneratorSRC.get_scaled_loss(totalGsrXYZLoss)
            if args.ganFlag:
                totalDsrXYLossScal = optimizerDiscriminatorSR.get_scaled_loss(dsrXYLoss)
                totalDsrYZLossScal = optimizerDiscriminatorSRC.get_scaled_loss(dsrYZLoss)
                
        # calculate gradients
        gradGsr = tape.gradient(totalGsrLossScal, generatorSR.trainable_variables)
        gradGsrc = tape.gradient(totalGsrcLossScal, generatorSRC.trainable_variables)
        if args.ganFlag:
            gradDsrXY = tape.gradient(totalDsrXYLossScal, discriminatorSR.trainable_variables)
            gradDsrYZ = tape.gradient(totalDsrYZLossScal, discriminatorSRC.trainable_variables)
            
        # unscale gradients
        gradGsr = optimizerGeneratorSR.get_unscaled_gradients(gradGsr)
        gradGsrc = optimizerGeneratorSRC.get_unscaled_gradients(gradGsrc)
        if args.ganFlag:
            gradDsrXY = optimizerDiscriminatorSR.get_unscaled_gradients(gradDsrXY)
            gradDsrYZ = optimizerDiscriminatorSRC.get_unscaled_gradients(gradDsrYZ)
            
        # apply gradients
        optimizerGeneratorSR.apply_gradients(zip(gradGsr,generatorSR.trainable_variables))
        optimizerGeneratorSRC.apply_gradients(zip(gradGsrc,generatorSRC.trainable_variables))
        if args.ganFlag:
            optimizerDiscriminatorSR.apply_gradients(zip(gradDsrXY,discriminatorSR.trainable_variables))
            optimizerDiscriminatorSRC.apply_gradients(zip(gradDsrYZ,discriminatorSRC.trainable_variables))
            
        return totalGsrXYLoss, totalGsrYZLoss, advsrXYLoss, dsrXYLoss, advsrYZLoss, dsrYZLoss

   
    @tf.function
    def distributed_train_step(HRBatch,BCBatch):
        PRGABL, PRGBAL, PRADVXYSRL, PRDXYSRL, PRADVYZSRL, PRDYZSRL = strategy.run(train_step, args=(HRBatch,BCBatch))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, PRGABL, axis=None),  strategy.reduce(tf.distribute.ReduceOp.SUM, PRGBAL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRADVXYSRL, axis=None),  strategy.reduce(tf.distribute.ReduceOp.SUM, PRDXYSRL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRADVYZSRL, axis=None),  strategy.reduce(tf.distribute.ReduceOp.SUM, PRDYZSRL, axis=None)
        
    # begin actual script here
    generatorSR, optimizerGeneratorSR = createSRGenerator(args)
    generatorSRC, optimizerGeneratorSRC = createSRCGenerator(args)
    discriminatorSR, optimizerDiscriminatorSR = createSRDiscriminator(args)
    discriminatorSRC, optimizerDiscriminatorSRC = createSRDiscriminator3D(args)
    
    trainingDir=f"./{args.checkpoint_dir}/{args.modelName}/"
    if args.continue_train or args.phase == 'test': # restore the weights if requested, or if testing
        print(f'Loading checkpoints from {trainingDir} for epoch {args.continueEpoch}')
        try:
            generatorSR.load_weights(f'{trainingDir}/GSR-{args.continueEpoch}/GSR')
            generatorSRC.load_weights(f'{trainingDir}/GSRC-{args.continueEpoch}/GSRC')
        except:
            print('Could not load SR related weights')
            
        if args.ganFlag:
            try:
                discriminatorSR.load_weights(f'{trainingDir}/DSR-{args.continueEpoch}/DSR')
                discriminatorSRC.load_weights(f'{trainingDir}/DSRC-{args.continueEpoch}/DSRC')
            except:
                print('Could not load SRGAN related weights')
    # run
    if args.phase == 'train':
        EPOCHS = args.epoch
        valoutDir = args.dataset_dir.split('/')[-2]
        # Create a checkpoint directory to store the checkpoints.
        rightNow=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trainOutputDir=f'./training_outputs/{rightNow}-distNN-{valoutDir}-{args.modelName}/'
        if not os.path.exists(trainingDir):
            os.mkdir(trainingDir)
        os.mkdir(trainOutputDir)

        print('2D/3D training specified, datasets will be randomly mini-batched per epoch')
        print('2D/3D dataset and training -> data will be fully preloaded into RAM')
        
        BCLoc=glob(args.dataset_dir+'LR/LR.npy')
        LRxy=np.load(BCLoc[0])
        #LRxy=np.transpose(LRxy,[2,1,0])

        
        HRLoc=glob(args.dataset_dir+'HR/HR.npy')
        HR=np.load(HRLoc[0])
        #HR=np.transpose(HR,[2,1,0])
        if args.valTest:
            LRTestLoc=glob(args.dataset_dir+'test/*')
            LRTest=np.load(LRTestLoc[0])
            LRTest=tf.cast(LRTest, tf.float32)
            LRTest=tf.expand_dims(LRTest,3)
        start_time = time.time()
        for epoch in range(EPOCHS):
            if args.ganFlag:
                batchSizeThisEpoch = args.batch_size
                fineSizeThisEpoch = args.fine_size
            else:
                totalPerBatchVoxels=args.fine_size*args.fine_size*args.batch_size
                minPerDimSize=args.scale*2
                maxPerDimSize=args.fine_size
                batchSizeThisEpoch =int(np.floor(np.random.rand()*(maxPerDimSize-minPerDimSize))+minPerDimSize)
                fineSizeThisEpoch = int(np.floor(np.sqrt(totalPerBatchVoxels/batchSizeThisEpoch)))
            print(f'Reading and Distributing Dataset into GPUs, block size this epoch: {batchSizeThisEpoch} x {fineSizeThisEpoch} x {fineSizeThisEpoch} -> {args.scale}x')
            realHRBatches, realBCBatches = createTrainingCubes2(args,HR,LRxy,batchSizeThisEpoch,fineSizeThisEpoch, args.scale)           
           
            HR_dataset = tf.data.Dataset.from_tensor_slices((realHRBatches)).batch(batchSizeThisEpoch*args.scale) 
            HR_dataset_dist = strategy.experimental_distribute_dataset(HR_dataset)
            
            HR_dataset_test=tf.data.Dataset.from_tensor_slices((realHRBatches[0:args.valNum*batchSizeThisEpoch*args.scale])).batch(batchSizeThisEpoch*args.scale) 
            
            
            LR_dataset = tf.data.Dataset.from_tensor_slices((realBCBatches)).batch(batchSizeThisEpoch) 
            LR_dataset_dist = strategy.experimental_distribute_dataset(LR_dataset)
            
            LR_dataset_test=tf.data.Dataset.from_tensor_slices((realBCBatches[0:args.valNum*batchSizeThisEpoch])).batch(batchSizeThisEpoch) 
            # TRAIN LOOP
            lastTime=time.time()

            lr=args.lr * 0.5**(epoch/args.epoch_step) # add cosine annealing later

            optimizerGeneratorSR.learning_rate = lr
            optimizerGeneratorSRC.learning_rate = lr
            totGABL = 0
            totGBAL = 0
            totADVXYSRL = 0
            totDXYSRL = 0
            totADVYZSRL = 0
            totDYZSRL = 0
            num_batches = 0
            numSkips=0;
            print(f'Learning Rate: {lr:.4e}')
            while num_batches < args.itersPerEpoch*args.iterCyclesPerEpoch:
                for x, y in zip(HR_dataset, LR_dataset):
                    num_batches += 1
                    GABL, GBAL, ADVXYSRL, DXYSRL, ADVYZSRL, DYZSRL = distributed_train_step(x, y)
                    totGABL += GABL
                    totGBAL += GBAL
                    totADVXYSRL += ADVXYSRL
                    totDXYSRL += DXYSRL
                    totADVYZSRL += ADVYZSRL
                    totDYZSRL += DYZSRL
                    currentTime=time.time()
                    
                    
                    stdout.write("\rEpoch: %4d, Iter: %4d, Time: %4.4f, Speed: %4.4f its/s, GSRxyL: %4.4f, GSRyzL: %4.4f, advSRxyL: %4.4f, advSRyzL: %4.4f, DSRxyL: %4.4f, DSRyzL: %4.4f" % (epoch+1, num_batches, currentTime-start_time, 1/(currentTime-lastTime), GABL, GBAL, ADVXYSRL, ADVYZSRL, DXYSRL, DYZSRL))
                    stdout.flush()
                    lastTime=currentTime

            stdout.write("\n")
            num_batches=num_batches-numSkips
            totGABL /= num_batches
            totGBAL /= num_batches
            totADVXYSRL /= num_batches
            totDXYSRL /= num_batches
            totADVYZSRL /= num_batches
            totDYZSRL /= num_batches
            print('Mean Epoch Performance: GSRxyL: %4.4f, GSRyzL: %4.4f, advSRxyL: %4.4f, advSRyzL: %4.4f, DSRxyL: %4.4f, DSRyzL: %4.4f' % (totGABL, totGBAL, totADVXYSRL, totADVYZSRL, totDXYSRL, totDYZSRL))
            
            if np.mod(epoch+1, args.print_freq) == 0 or epoch == 0:
                # validation LOOP

                valPSNRC=0.0
                valPSNRCC=0.0
                
                numTestBatches=0
                os.mkdir(f'./{trainOutputDir}/epoch-{epoch+1}/')

                for C, B in zip(HR_dataset_test, LR_dataset_test):

                    #B = BC[0][1]
                    #C = BC[0][0]

                    Cd = tf.image.resize(tf.squeeze(C),[C.shape[0]//args.scale,C.shape[2]],method='bicubic')
                    Cd=tf.expand_dims(Cd,3)
                    Co = np.asarray(Cd)
                    fakeC = generatorSR(B, training=False)
                    fakeCo = np.asarray(fakeC)
                    
                    psnrC=tf.image.psnr(fakeC,Cd,2)
                    # set bit depth to 8 for SRxy
                    fakeC=(fakeC+1)*127.5
                    fakeC=tf.math.round(fakeC)
                    fakeC=fakeC/127.5 - 1
                    # transpose and downsample here
                    fakeC = tf.transpose(fakeC,[1,0,2,3])
                    B = tf.transpose(B,[1,0,2,3])
                    C = tf.transpose(C,[1,0,2,3])
                    #fakeC=tf.image.resize(fakeC,[fakeC.shape[1],fakeC.shape[2]//args.scale],method='bicubic')
                    fakeC_clean = generatorSRC(fakeC, training=False)
                    psnrCC=tf.image.psnr(fakeC_clean,C,2)
                    
                    B = np.asarray(B)
                    C = np.asarray(C)
                    fakeC = np.asarray(fakeC)
                    fakeC_clean = np.asarray(fakeC_clean)

                    valPSNRC += np.mean(psnrC)
                    valPSNRCC += np.mean(psnrCC)
                    numTestBatches += 1

                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Bxy.tif'
                    B=(B+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(B.astype('uint8')), dtype='uint8'))

                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Cxyz.tif'
                    Co=(Co+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(Co.astype('uint8')), dtype='uint8'))
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Ctxyz.tif'
                    C=(C+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(C.astype('uint8')), dtype='uint8'))

                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxy.tif'
                    fakeCo=(fakeCo+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(fakeCo.astype('uint8')), dtype='uint8'))

                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxytd.tif'
                    fakeC=(fakeC+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(fakeC.astype('uint8')), dtype='uint8'))
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxyz.tif'
                    fakeC_clean=(fakeC_clean+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(fakeC_clean.astype('uint8')), dtype='uint8'))
                        

                    
                    stdout.write("\rIter: %4d, Test: PSNR-SR: %4.4f, PSNR-SRC: %4.4f" %(numTestBatches, np.mean(psnrC), np.mean(psnrCC)))
                    stdout.flush()
                    if numTestBatches == args.valNum:
                        break

                valPSNRC /= numTestBatches
                valPSNRCC /= numTestBatches

                stdout.write("\n")
                print(f'Mean Validation PSNR-SR: {valPSNRC}, PSNR-SRC: {valPSNRCC}')
                
                if args.valTest:
                    print(f'Generating some test cubes')
                    testSRxy=generatorSR(LRTest)
                    testSRxy = np.asarray(testSRxy)
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/testSRxy.tif'
                    testSRxy=(testSRxy+1)*127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(testSRxy.astype('uint8')), dtype='uint8'))
                        
            if (epoch) % args.save_freq == 0:
                #checkpoint.save(checkpoint_prefix)
                print('Saving network weights (archive)')
                generatorSR.save_weights(f'{trainingDir}/GSR-{epoch}/GSR')
                generatorSRC.save_weights(f'{trainingDir}/GSRC-{epoch}/GSRC')
                if args.ganFlag:
                    discriminatorSR.save_weights(f'{trainingDir}/DSR-{epoch}/DSR')
                    discriminatorSRC.save_weights(f'{trainingDir}/DSRC-{epoch}/DSRC')
                        
                print('Saving network weights (rewritable checkpoint)')
                generatorSR.save_weights(f'{trainingDir}/GSR/GSR')
                generatorSRC.save_weights(f'{trainingDir}/GSRC/GSRC')
                if args.ganFlag:
                    discriminatorSR.save_weights(f'{trainingDir}/DSR/DSR')
                    discriminatorSRC.save_weights(f'{trainingDir}/DSRC/DSRC')
                    
                print('Saving model (rewritable checkpoint)')
                generatorSR.save(f'{trainingDir}/GSR-{epoch}.h5')
                generatorSRC.save(f'{trainingDir}/GSRC-{epoch}.h5')


    elif args.phase == 'testSmall':
        # test within scope?
        # read entire LR block of size x,y,zb and upscale to xs,ys,zs

        testFiles = sorted(glob(args.test_dir+'/*.npy'))          
            
        i=0
        for testFile in testFiles:
            #testFile=testFiles[0]
            print(f'XY Pass: Super Resolving {testFile}')
            
            domain=np.load(testFile)
            domainSRxy=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]],'uint8')
            domainSRxyz=np.zeros([domain.shape[0]*args.scale,domain.shape[1]*args.scale,domain.shape[2]*args.scale],'uint8')
            for z in range(domain.shape[2]):
                print(f'XY Pass: Super Resolving slice {z}')
                slicez=domain[:,:,z]
                slicez = (slicez/127.5) - 1 # block will auto cast to float, thanks python
                slicez=tf.cast(slicez, tf.float32)
                slicez=tf.expand_dims(slicez,2)
                slicez=tf.expand_dims(slicez,0)
                ABsr=generatorSR(slicez)
                ABsr=(ABsr+1)*127.5
                ABsr=tf.math.round(ABsr)
                ABsr=np.asarray(ABsr,'uint8')
                #image_path = f'{testFile}-SRxy-{z}-{args.scale}x-{args.modelName}.tif'
                #tifffile.imwrite(image_path, np.squeeze(ABsr))

                domainSRxy[:,:,z]=ABsr[0,:,:,0]
                
            for x in range(domainSRxy.shape[0]):
                print(f'YZ Pass: Super Resolving slice {x}')
                slicex=domainSRxy[x,:,:]
                slicex=slicex/127.5 - 1
                slicex=tf.transpose(slicex,[1,0])
                slicex=tf.expand_dims(slicex,2)
                slicex=tf.expand_dims(slicex,0)
                ABsr=generatorSRC(slicex)
                ABsr=np.squeeze(ABsr)
                ABsr=tf.transpose(ABsr,[1,0])
                ABsr=(ABsr+1)*127.5
                domainSRxyz[x,:,:]=np.asarray(ABsr,'uint8')
            domainSRxyz=np.transpose(domainSRxyz,[2,0,1])
            image_path = f'{testFile}-SRxyz-{args.scale}x-{args.modelName}.tif'
            tifffile.imwrite(image_path, np.array(domainSRxyz))

    elif args.phase == 'test':
        # test within scope?
        # read entire LR block of size x,y,zb and upscale to xs,ys,zs

        testFiles = sorted(glob(args.test_dir+'/*.png'))
        
        if not os.path.exists(f'{args.test_save_dir}/{args.modelName}'):
            os.mkdir(f'{args.test_save_dir}/{args.modelName}')
            
        if not os.path.exists(f'{args.test_temp_save_dir}/{args.modelName}'):
            os.mkdir(f'{args.test_temp_save_dir}/{args.modelName}')
            
            
        i=0
        for testFile in testFiles:
            #testFile=testFiles[0]
            print(f'XY Pass: Super Resolving {testFile}')
            slicez=imageio.imread(testFile)
            slicez = (slicez/127.5) - 1 # block will auto cast to float, thanks python
            #slicez=slicez[500:600,500:600]
            #blocksrxyz=tf.zeros([1, block.shape[0]*args.scale, block.shape[1]*args.scale, block.shape[2]])
            fileName=testFile.split('.')[0]
            fileName=fileName.split('/')[-1]
            slicez=tf.cast(slicez, tf.float32)
            slicez=tf.expand_dims(slicez,2)
            slicez=tf.expand_dims(slicez,0)
            
            maxNz=1000
            dualLength=100
            numParts=slicez.shape[2]//(maxNz-dualLength)
            z=0
            zz=0
            maxSR=np.zeros([slicez.shape[1]*args.scale,slicez.shape[2]*args.scale],'uint8')
            if numParts==0:
                print(f'Super Resolving Whole Slice')
                tempSlice=slicez
                ABsr=generatorSR(tempSlice)
                ABsr=np.asarray(ABsr)
                ABsr=np.squeeze(ABsr)
                ABsr=(ABsr+1)*127.5
                ABsr=tf.math.round(ABsr)
                ABsr=np.asarray(ABsr,'uint8')
                maxSR=ABsr
            for n in range(numParts):
                print(f'Super Resolving Subsection {n+1}')
                tempSlice=slicez[:,:,zz:zz+maxNz]
                ABsr=generatorSR(tempSlice)
                ABsr=np.asarray(ABsr)
                ABsr=np.squeeze(ABsr)
                ABsr=(ABsr+1)*127.5
                ABsr=tf.math.round(ABsr)
                ABsr=np.asarray(ABsr,'uint8')
                if n==0:
                    maxSR[:,:(z+maxNz-dualLength//2)*args.scale]=ABsr[:,:(maxNz-dualLength//2)*args.scale]
                    z=z+maxNz-dualLength//2
                elif n==numParts-1:
                    maxSR[:,(z)*args.scale:]=ABsr[:,dualLength//2*args.scale:]
                else:
                    maxSR[:,(z)*args.scale:(z+maxNz-dualLength)*args.scale]=ABsr[:,dualLength//2*args.scale:(maxNz-dualLength//2)*args.scale]
                    z=z+maxNz-dualLength
                zz=zz+maxNz-dualLength

            for j in range(maxSR.shape[1]):
                np.save(f'{args.test_temp_save_dir}/{args.modelName}/{fileName}_result_SRxy_{i}_{j}.npy',maxSR[:,j]) 
                stdout.write("\rSaving stick %d" % (j+1))
                stdout.flush()
            stdout.write("\n")
            if np.mod(i,100)==0:
                imageio.imwrite(f'{args.test_temp_save_dir}/{fileName}_result_SRxy_{i}.png', maxSR.astype(np.uint8))
            i=i+1

#        stacks=[] # dont initialise to fool python into paging the slices - after above loop to reduce error time
#        for z in range(len(testFiles)):
#            testFile=testFiles[z]
#            fileName=testFile.split('.')[0]
#            fileName=fileName.split('/')[-1]
#            slicez = np.load(f'{args.test_temp_save_dir}/{args.modelName}/{fileName}_result_SRxy_{z}.npy')
#            stacks.append(slicez)
#            stdout.write("\rLoading XY Slice %d" % (z+1))
#            stdout.flush()
#        stdout.write("\n")
#        print(f'Stack Loaded')
        # transpose the stack in pieces. I guess....
        #stacks=np.stack(stacks,2)
        #ABsr=np.zeros([5688,5688])
        for j in range(22751, 32400):
            #ABsr=np.zeros([5688,5688])
            print(f'XZ Pass: Downsampling and Super Resolving Slice {j}')
            transSlice=np.zeros([len(testFiles),maxSR.shape[0]],'uint8')
            for i in range(len(testFiles)):
                testFile=testFiles[i]
                fileName=testFile.split('.')[0]
                fileName=fileName.split('/')[-1]
                transSlice[i,:]=np.load(f'{args.test_temp_save_dir}/{args.modelName}/{fileName}_result_SRxy_{i}_{j}.npy') 
                stdout.write("\rLoading XZ Slice %d" % (i+1))
                stdout.flush()
            stdout.write("\n")
#            
            transSlice=tf.cast(transSlice, tf.float32)
            transSlice=transSlice/127.5 - 1
            transSlice=tf.expand_dims(transSlice,2)
            #transSlice=tf.image.resize(transSlice,[transSlice.shape[0]//args.scale,transSlice.shape[1]],method='bicubic')
            print(f'Super Resolving')
            transSlice=tf.expand_dims(transSlice,0)

            maxNz=10000
            dualLength=100
            numParts=transSlice.shape[2]//(maxNz-dualLength)
            z=0
            zz=0
            maxSR=np.zeros([transSlice.shape[1]*args.scale,transSlice.shape[2]*args.scale],'uint8')
            if numParts==0:
                print(f'Super Resolving Whole Slice')
                tempSlice=transSlice
                ABsr=generatorSRC(tempSlice)
                ABsr=np.asarray(ABsr)
                ABsr=np.squeeze(ABsr)
                ABsr=(ABsr+1)*127.5
                ABsr=tf.math.round(ABsr)
                ABsr=np.asarray(ABsr,'uint8')
                maxSR=ABsr
            
            for n in range(numParts):
                print(f'Super Resolving Subsection {n+1}')
                tempSlice=transSlice[:,:,zz:zz+maxNz]
                ABsr=generatorSRC(tempSlice)
                ABsr=np.asarray(ABsr)
                ABsr=np.squeeze(ABsr)
                ABsr=(ABsr+1)*127.5
                ABsr=tf.math.round(ABsr)
                ABsr=np.asarray(ABsr,'uint8')
                if n==0:
                    maxSR[:,:(z+maxNz-dualLength//2)*args.scale]=ABsr[:,:(maxNz-dualLength//2)*args.scale]
                    z=z+maxNz-dualLength//2
                elif n==numParts-1:
                    maxSR[:,(z)*args.scale:]=ABsr[:,dualLength//2*args.scale:]
                else:
                    maxSR[:,(z)*args.scale:(z+maxNz-dualLength)*args.scale]=ABsr[:,dualLength//2*args.scale:(maxNz-dualLength//2)*args.scale]
                    z=z+maxNz-dualLength
                zz=zz+maxNz-dualLength
#            for j in range(ABsr.shape[0]):
#                np.save(f'{args.test_temp_save_dir}/{args.modelName}/{fileName}_result_SRxy_{i}_{j}.npy',ABsr[:,j]) 
#                stdout.write("\rSaving stick %d" % (j+1))
#                stdout.flush()
#            stdout.write("\n")
#            if np.mod(i,100)==0:
#                imageio.imwrite(f'{args.test_temp_save_dir}/{fileName}_result_SRxy_{i}.png', ABsr.astype(np.uint8))
            #tifffile.imwrite(f'{args.test_save_dir}/{args.modelName}/{fileName}_result_SRxyz_{j}.tif', maxSR)
            imageio.imwrite(f'{args.test_save_dir}/{args.modelName}/{fileName}_result_SRxyz_{j}.png', maxSR)
#            ABsr=generatorSRC(transSlice)
#            ABsr=np.asarray(ABsr)
#            ABsr=np.squeeze(ABsr)
#            ABsr=(ABsr+1)*127.5
#            ABsr=tf.math.round(ABsr)
#            ABsr=np.asarray(ABsr,'uint8')
#            imageio.imwrite(f'{args.test_save_dir}/{args.modelName}/{fileName}_result_SRxyz_{j}.png', ABsr)


