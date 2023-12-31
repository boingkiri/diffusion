import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import os
import scipy
from tqdm import tqdm


def compute_statistics(path, params, apply_fn, batch_size=1, img_size=None, model=None):
    if path.endswith(".npz"):
        stats = np.load(path)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma

    images = []
    for f in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, f))
        # convert if only a single channel
        if img.mode == "L":
            img = img.convert("RGB")
        # resize if not the right size
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images.append(img)

    num_batches = int(len(images) // batch_size)
    act = []
    for i in tqdm(range(num_batches)):
        x = images[i * batch_size : i * batch_size + batch_size]
        x = np.asarray(x)
        x = 2 * x - 1
        pred = apply_fn(params, jax.lax.stop_gradient(x))
        act.append(pred.squeeze(axis=1).squeeze(axis=1))
    act = jnp.concatenate(act, axis=0)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def compute_mean_difference(mu1, mu2):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    diff = mu1 - mu2
    return diff.dot(diff)


def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean