import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from translator import GPPatchMcResDis, FewShotGen


# TODO: in config lr_gen, lr_dis

def recon_criterion(predict, target):
    return tf.math.reduce_mean(tf.math.abs(predict - target))  # torch.mean(torch.abs(predict - target))


class Trainer:
    def __init__(self, config):
        self.config = config
        lr_gen = config['lr_gen']
        lr_dis = config['lr_dis']

        self.gen_opt = RMSprop(learning_rate=lr_gen)
        self.dis_opt = RMSprop(learning_rate=lr_dis)

        self.generator = FewShotGen(config)
        self.discriminator = GPPatchMcResDis(config)

    def train(self, co_data, cl_data, config, epochs):
        for i in range(epochs):
            dis_l_total = dis_update(self, co_data, cl_data, config)
            gen_l_total = gen_update(self, co_data, cl_data, config)

            print('D acc: %.4f\t G acc: %.4f' % (dis_l_total, gen_l_total))

    def dis_update(self, co_data, cl_data, config):
        xa = co_data[0]
        la = co_data[1]
        xb = cl_data[0]
        lb = cl_data[1]
        # xb.requires_grad_()  # TODO change to new training (Begins to watch changes to this tensor)
        with tf.GradientTape() as tape:
            l_real_pre, acc_r, resp_r = self.discriminator.calc_dis_real_loss(xb, lb)
            l_real = config['gan_w'] * l_real_pre
            l_real.backward(retain_graph=True)  # TODO change to new training
            l_reg_pre = self.discriminator.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()  # TODO change to new training
        with torch.no_grad():  # TODO change to new training
            c_xa = self.gen.enc_content(xa)
            s_xb = self.gen.enc_class_model(xb)
            xt = self.gen.decode(c_xa, s_xb)
        l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(), lb) # TODO change to new training (detatches Tensor from current graph)
        l_fake = config['gan_w'] * l_fake_p
        l_fake.backward()  # TODO change to new training
        l_total = l_fake + l_real + l_reg
        acc = 0.5 * (acc_f + acc_r)
        return l_total

    def gen_update(self, co_data, cl_data, config):
        xa = co_data[0]
        la = co_data[1]
        xb = cl_data[0]
        lb = cl_data[1]
        with tf.GradientTape() as tape:
            c_xa = self.generator.enc_content(xa)
            s_xa = self.generator.enc_class_model(xa)
            s_xb = self.generator.enc_class_model(xb)
            xt = self.generator.decode(c_xa, s_xb)  # translation
            xr = self.generator.decode(c_xa, s_xa)  # reconstruction
            l_adv_t, gacc_t, xt_gan_feat = self.discriminator.calc_gen_loss(xt, lb)
            l_adv_r, gacc_r, xr_gan_feat = self.discriminator.calc_gen_loss(xr, la)
            _, xb_gan_feat = self.discriminator(xb, lb)
            _, xa_gan_feat = self.discriminator(xa, la)
            l_c_rec = recon_criterion(tf.math.reduce_mean(tf.math.reduce_mean(xr_gan_feat, axis=3), axis=2),
                                      tf.math.reduce_mean(tf.math.reduce_mean(xa_gan_feat, axis=3), axis=2))
            l_m_rec = recon_criterion(tf.math.reduce_mean(tf.math.reduce_mean(xt_gan_feat, axis=3), axis=2),
                                      tf.math.reduce_mean(tf.math.reduce_mean(xb_gan_feat, axis=3), axis=2))
            l_x_rec = recon_criterion(xr, xa)
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (config['gan_w'] * l_adv + config['r_w'] * l_x_rec + config['fm_w'] * (l_c_rec + l_m_rec))
        gradients = tape.gradient(l_total, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return l_total
