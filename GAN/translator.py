import tensorflow as tf
import tensorflow_transform as tft

class AdaptiveInstanceNorm2d(tf.keras.Model):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

def get_functions(norm='none', activation='none', pad_type='none'): 
    # initialize padding
    if pad_type == 'reflect':
        # self.pad = nn.ReflectionPad2d(padding)
        assert 0, "Unsupported padding type: {}".format(pad_type)
    elif pad_type == 'replicate':
        # self.pad = nn.ReplicationPad2d(padding)
        assert 0, "Unsupported padding type: {}".format(pad_type)
    elif pad_type == 'zero':
        use_pad = tf.keras.layers.ZeroPadding2D(padding=(padding,padding))
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

     # initialize normalization
    norm_dim = out_dim
    if norm == 'bn':
        use_norm = tf.keras.layers.BatchNormalization()
    elif norm == 'in':
        use_norm = tf.keras.layers.BatchNormalization(axis=[0,1]) # This should be Instance Normalisation
    elif norm == 'adain':
        use_norm = AdaptiveInstanceNorm2d(norm_dim)
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
    def __init__(self, out_dim, ks, st, padding = 0,
                 norm = 'none', activation='relu', pad_type = 'zero',
                 use_bias=True, activation_first = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        self.norm, self.activation, self.pad  = get_functions(norm, activation, pad_type)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
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


class ResBlocks(tf.keras.Model):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type = pad_type)]
        self.model = tf.keras.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(tf.keras.Model):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = tf.keras.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class LinearBlock(tf.keras.Model):
    def __init__(self, out_dim, norm='none', activation = 'relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = tf.keras.layers.Dense(out_dim)
        self.norm, self.activation, _ = get_functions(norm, activation, pad_type)

    def forward(self, x):
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
        self.model = tf.keras.Sequential(*sel.model)
        self.output_dim = dim
    
    def forward(self, x):
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
        self.model += [tf.keras.layers.GlobalAveragePooling2D()] # = nn.AdaptiveAvgPool2d(1)?
        self.model += [nn.Conv2D(dim, latent_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim
    
    def forward(self, x):
        return self.model(x)

class Decoder(tf.keras.Model):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()
        
        self.model = []
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
        self.model = tf.keras.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(tf.keras.Model):
    def __init__(self, out_dim, dim, n_blk, norm, activ):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim, norm='none', activation='none')]
        self.model = tf.keras.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# def assign_adain_params(adain_params, model):

        
class FewShotGen(tf.keras.Model):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']
        self.enc_class_model = ClassModelEncoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
        self.enc_content = ContentEncoder(down_content, n_res_blks, nf, norm='in', activ='relu', pad_type='reflect')
        self.dec = Decoder(down_content, n_res_blks, self.enc_content.output_dim, 3, res_norm='adain', activ='relu', pad_type='refelect')
        self.mlp = MLP(get_num_adain_params(self.dec), nf_mlp, n_mlp_blks, norm='none', activ='relu')
    
    def forward(self, content_image, style_set):
        content, model_codes = self.encode(content_image, style_set)
        model_code = tft.mean(model_codes, reduce_instance_dims=False) # torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, content_image, style_set):
        content = self.enc_content(content_image)
        class_codes = self.enc_class_model(style_set)
        class_code = tft.mean(class_codes, reduce_instance_dims=False) # torch.mean(model_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, model_code):
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images