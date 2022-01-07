import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def flatten(l):
    r = []
    for sl in l:
        if sl:
            for i in sl:
                r.append(i)
    return r

class AdaptiveInstanceNorm2d(tf.keras.layers.Layer):
    def __init__(self, num_features, name, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__(name=name)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.eps)

    def get_mean_std(x, epsilon=1e-5): # From https://keras.io/examples/generative/adain/
        axes = [1, 2]

        # Compute the mean and standard deviation of a tensor.
        mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
        standard_deviation = tf.sqrt(variance + epsilon)
        return mean, standard_deviation
    
    def ada_in(self, x):
        x_mean, x_std = get_mean_std(x)
        return self.weight * ((x - x_mean) / x_std) + self.bias

    def call(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        x = self.batch_norm(x)
        x = self.ada_in(x)
        return x

class ReflectionPadding2D(tf.keras.layers.Layer): # From https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

class ReplicationPadding2D(tf.keras.layers.Layer): # From https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')

def get_functions(norm='none', activation='none', pad_type='none', adain_name=None, padding=None, out_dim=None): 
    # initialize padding
    if pad_type == 'reflect':
        # self.pad = nn.ReflectionPad2d(padding)
        use_pad = ReflectionPadding2D(padding=(padding, padding))
    elif pad_type == 'replicate':
        # self.pad = nn.ReplicationPad2d(padding)
        use_pad = ReplicationPadding2D(padding=(padding, padding))
    elif pad_type == 'zero':
        use_pad = tf.keras.layers.ZeroPadding2D(padding=(padding,padding))
    elif pad_type == 'none':
        use_pad = None
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

     # initialize normalization
    norm_dim = out_dim
    if norm == 'bn':
        use_norm = tf.keras.layers.BatchNormalization()
    elif norm == 'in':
        use_norm = tf.keras.layers.BatchNormalization(axis=[0,1]) # This should be Instance Normalisation
    elif norm == 'adain':
        use_norm = AdaptiveInstanceNorm2d(norm_dim, adain_name)
    elif norm == 'none':
        use_norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
        
    # initialize activation
    if activation == 'relu':
        use_activation = tf.keras.layers.ReLU()
    elif activation == 'lrelu':
        use_activation = tf.keras.layers.LeakyReLU(0.2)
    elif activation == 'tanh':
        use_activation = Tanh()
    elif activation == 'none':
        use_activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)

    return use_norm, use_activation, use_pad


class Tanh(tf.keras.layers.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def call(self, inputs):
        return tf.keras.activations.tanh(inputs)

class Conv2dBlock(tf.keras.Model):
    def __init__(self, out_dim, ks, st, padding = 0, adain_name='',
                 norm = 'none', activation='relu', pad_type = 'zero',
                 use_bias=True, activation_first = False):
        assert adain_name if norm == 'adain' else True, "If adain is used as norm, adain name must be given."
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        self.norm_name = norm
        self.norm, self.activation, self.pad  = get_functions(norm, activation, pad_type, adain_name=adain_name, padding=padding, out_dim=out_dim)

        self.conv = tf.keras.layers.Conv2D(out_dim, ks, strides=(st,st), use_bias=self.use_bias)

    def call(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x

    def get_adain_layers(self):
        if self.norm_name == 'adain':
            return self.norm
        else:
            return None        

class ResBlocks(tf.keras.Model):
    def __init__(self, num_blocks, dim, norm, activation, pad_type, adain_names=[]):
        assert len(adain_names) == num_blocks if norm == 'adain' else True, "If adain is used as norm, enough adain names must be given."
        super(ResBlocks, self).__init__()
        self.model = []
        self.adain_blocks = []
        if norm == 'adain':
            for n in adain_names:
                self.adain_blocks += [ResBlock(dim,
                                        norm=norm,
                                        activation=activation,
                                        pad_type = pad_type,
                                        adain_name=n)]
            self.model += self.adain_blocks
        else:
            for i in range(num_blocks):
                self.model += [ResBlock(dim,
                                        norm=norm,
                                        activation=activation,
                                        pad_type = pad_type)]
        self.model = tf.keras.Sequential(self.model)

    def call(self, x):
        return self.model(x)

    def get_adain_layers(self):
        return flatten([b.get_adain_layers() for b in self.adain_blocks])

class ResBlock(tf.keras.Model):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', adain_name=""):
        super(ResBlock, self).__init__()
        self.adain_blocks = []
        model = []
        model += [Conv2dBlock(dim, 3, 1, 1,
                              adain_name=adain_name,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, 3, 1, 1,
                              adain_name=adain_name,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        if norm == 'adain':
            self.adain_blocks = model
        self.model = tf.keras.Sequential(model)

    def call(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

    def get_adain_layers(self):
        return [b.get_adain_layers() for b in self.adain_blocks]

class ActFirstResBlock(tf.keras.Model):
    def __init__(self, fin, fout, fhid=None, activation='lrelu', norm='none'):
        super(ActFirstResBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def call(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out

class LinearBlock(tf.keras.Model):
    def __init__(self, out_dim, norm='none', activation = 'relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = tf.keras.layers.Dense(out_dim)
        self.norm, self.activation, _ = get_functions(norm, activation, out_dim=out_dim)

    def call(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class ContentEncoder(tf.keras.Model):
    def __init__(self, downs, n_res, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)]
        self.model = tf.keras.Sequential(self.model)
        self.output_dim = dim
    
    def call(self, x):
        return self.model(x)

class ClassModelEncoder(tf.keras.Model):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [tf.keras.layers.AveragePooling2D(pool_size=(1,1))] # = nn.AdaptiveAvgPool2d(1)?
        self.model += [tf.keras.layers.Conv2D(latent_dim, 1, strides=(1,1))]
        self.model = tf.keras.Sequential(self.model)
        self.output_dim = dim
    
    def call(self, x):
        return self.model(x)

class Decoder(tf.keras.Model):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type, adain_names=None):
        super(Decoder, self).__init__()
        self.adain_blocks = []
        
        self.model = []
        if res_norm == 'adain':
            self.adain_blocks = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, adain_names=adain_names)
            self.model += [self.adain_blocks]
        else:
            self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]

        for i in range(ups):
            self.model += [tf.keras.layers.UpSampling2D(),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm = 'in',
                                       activation = activ,
                                       pad_type = pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = tf.keras.Sequential(self.model)

    def call(self, x):
        return self.model(x)

    def get_adain_layers(self):
        return self.adain_blocks.get_adain_layers()

class MLP(tf.keras.Model):
    def __init__(self, out_dim, dim, n_blk, norm, activ):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(out_dim, norm='none', activation='none')]
        self.model = tf.keras.Sequential(self.model)

    def call(self, x):
        return self.model(x.view(x.size(0), -1))

def assign_adain_params(adain_params, model, adain_names):
    for l in model.get_adain_layers():
        mean = adain_params[:, :l.num_features]
        std = adain_params[:, l.num_features:2*l.num_features]
        m.bias = tf.reshape(mean, [-1])
        m.weight = tf.reshape(std, [-1])
        if adain_params.size(1) > 2*l.num_features:
            adain_params = adain_params[:, 2*l.num_features:]

def get_num_adain_params(model, adain_names):
    num_adain_params = 0
    for l in model.get_adain_layers():
        num_adain_params += 2*l.num_features
    return num_adain_params
        
class FewShotGen(tf.keras.Model):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        self.adain_names = ["adain" + str(x) for x in range(n_res_blks)]
        latent_dim = hp['latent_dim']
        self.enc_class_model = ClassModelEncoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
        self.enc_content = ContentEncoder(down_content, n_res_blks, nf, norm='in', activ='relu', pad_type='reflect')
        self.dec = Decoder(down_content, n_res_blks, self.enc_content.output_dim, 3, res_norm='adain', activ='relu', pad_type='reflect', adain_names=self.adain_names)
        self.mlp = MLP(get_num_adain_params(self.dec, self.adain_names), nf_mlp, n_mlp_blks, norm='none', activ='relu')
    
    def call(self, content_image, style_set):
        content, model_codes = self.encode(content_image, style_set)
        model_code = tf.math.reduce_mean(model_codes, axis=0) # torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, content_image, style_set):
        content = self.enc_content(content_image)
        class_codes = self.enc_class_model(style_set)
        class_code = tf.math.reduce_mean(class_codes, axis=0) # torch.mean(model_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, model_code):
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.dec, self.adain_names)
        images = self.dec(content)
        return images

class GPPatchMcResDis(tf.keras.Model):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [ReflectionPadding2D(padding=(1,1))]
            cnn_f += [tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2))] # nn.AvgPool2d(kernel_size=3, stride=2) ?
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(hp['num_classes'], 1, 1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = tf.keras.Sequential(cnn_f)
        self.cnn_c = tf.keras.Sequential(cnn_c)

    def call(self, x, y):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        # index = torch.LongTensor(range(out.size(0))).cuda()
        # out = out[index, y, :, :]
        out = out[:, y, :, :]
        return out, feat

    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = np.prod(resp_fake.size())
        fake_loss = tf.keras.layers.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = np.prod(resp_real.size())
        real_loss = tf.keras.layers.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = np.prod(resp_fake.size())
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat