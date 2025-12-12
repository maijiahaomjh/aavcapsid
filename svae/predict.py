"""Predictions module for SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
"""

import pandas as pd
import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()

""" 
# -- predict --
Description: given a trained (supervised) VAE model and (one-hot encoded) inputs X, encodes X
into latent representation (z0, z1, ...) and makes regressor label predictions y_pred.
If true labels Y (optional argument) are specified, predictions are checked against Y
and the mean squared error (MSE) of predictions against true label is printed out.
Outputs a dataframe containing latent encodings, predicted labels, and true label values
if specified. Can optionally specify list of variants AA, in amino acid string form, as
an index for dataframe. Optionally writes this dataframe to file if a full output path
is specified. 
"""


def predict(model, X, Y=None, AA=None, outpath=None):

    encoder = model.get_layer("encoder")
    regressor = model.get_layer("regressor")

    # Get latent encoding and compressed features for regressor input
    # Encoder outputs: [z_mean, z_log_var, z, x_feat]
    enc_outputs = encoder.predict(X)
    z_encoded_dataset = enc_outputs[0]
    
    # 获取压缩特征 (index 3) 或回退到原始输入 X
    if len(enc_outputs) >= 4:
        x_feat = enc_outputs[3]
    else:
        x_feat = X
        
    # 自适应构建 Regressor 输入
    # 检查 Regressor 期望的输入维度
    expected_dim = regressor.input_shape[1]
    
    latent_dim = z_encoded_dataset.shape[1]
    feat_dim = x_feat.shape[1]
    
    if expected_dim == feat_dim:
        # 仅使用特征 (Latent 被忽略)
        regressor_input = x_feat
    elif expected_dim == latent_dim + feat_dim:
        # 使用 Latent + 特征
        regressor_input = tf.concat([z_encoded_dataset, x_feat], 1)
    else:
        print(f"Warning: Regressor expected input dim {expected_dim}, but z({latent_dim}) + x({feat_dim}) = {latent_dim+feat_dim}. Trying concat...")
        regressor_input = tf.concat([z_encoded_dataset, x_feat], 1)

    # Make regressor predictions
    preds = regressor.predict(regressor_input)

    # Dataframe to store latent encoding, regressor predictions, and true labels (if
    # specified), indexed by AA strings (if specified)
    preds_df = pd.DataFrame(preds.flatten(), columns=["y_pred"])

    # Add columns in preds_df for latent space coordinates
    for i in range(z_encoded_dataset.shape[1]):
        preds_df["z{}".format(i)] = z_encoded_dataset[:, i]

    # Add columns for compressed features (if available)
    if x_feat is not None:
        for i in range(x_feat.shape[1]):
            preds_df[f"c{i}"] = x_feat[:, i]

    # Add optional AA index
    if AA is not None:
        preds_df.insert(0, "AA_sequence", AA)

    # If Y specified, compute MSE of predictions vs true labels and add Y column to df
    if Y is not None:
        mse_y = mse(Y, preds.flatten()).numpy()
        print("\nMSE of predictions vs true labels: {}".format(mse_y))
        preds_df["y_true"] = Y

    # Write dataframe to file, if full path is specified
    if outpath is not None:
        preds_df.to_csv(outpath)

    return preds_df
