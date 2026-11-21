"""Explicit, shared evaluation protocols; none silently imitates paper table scores."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pixels(x):
    return expit(np.asarray(x, dtype=np.float64))


def w1_euclidean(x, y):
    """Exact empirical W1, equal sample weights/count, ground cost Euclidean (not W2)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError('W1 requires equal sample counts; select a fixed subset explicitly.')
    cost = cdist(x.reshape(len(x), -1), y.reshape(len(y), -1), metric='euclidean')
    i, j = linear_sum_assignment(cost)
    return float(cost[i, j].mean())


def vendi_cosine(x):
    """exp(entropy(eigenvalues(K/n))); K is PSD cosine Gram, unit diagonal."""
    a = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    zero = norm[:, 0] < 1e-12
    a = a / np.maximum(norm, 1e-12)
    k = a @ a.T
    k[np.diag_indices(len(k))] = 1.0  # zero samples get orthogonal unit features
    vals = np.maximum(np.linalg.eigvalsh(k / len(k)), 0)
    vals = vals[vals > 1e-14]
    return float(np.exp(-(vals * np.log(vals)).sum()))


def paired_ssim(x, pred):
    from skimage.metrics import structural_similarity
    if x.shape != pred.shape:
        raise ValueError('SSIM requires paired arrays of identical shape.')
    return float(np.mean([structural_similarity(a.squeeze(), b.squeeze(), data_range=1.0)
                          for a, b in zip(x, pred)]))


def soft_nmi_variants(probabilities, labels):
    """Separate the printed formula, standard arithmetic NMI, and released legacy formula."""
    from scipy.stats import entropy
    q=np.asarray(probabilities,dtype=float);y=np.asarray(labels,dtype=int)
    q=q/q.sum(1,keepdims=True)
    onehot=np.eye(q.shape[1])[y]
    joint=onehot.T@q/len(q)
    hy=entropy(joint.sum(1));hq=entropy(joint.sum(0));hj=entropy(joint.ravel())
    mi=max(0.,hy+hq-hj);den=hy+hq
    legacy_y=np.mean(entropy(onehot+1e-10,axis=0));legacy_q=np.mean(entropy(q+1e-10,axis=0))
    legacy_joint=-np.sum(joint*np.log(joint+1e-10));legacy_den=legacy_y+legacy_q
    return dict(soft_nmi_arithmetic_corrected=float(2*mi/den) if den else 0.,
                soft_nmi_paper_printed=float(mi/den) if den else 0.,
                soft_nmi_legacy=float(np.clip((legacy_den-legacy_joint)/legacy_den,0,1)) if legacy_den else 0.)


def evaluate_arrays(*, true_x, generated=None, posterior_pred=None, shuffled_pred=None,
                    train_z=None, train_y=None, test_z=None, test_y=None,
                    dataset='pinwheel', n_clusters=None, seed=2026):
    """Image inputs are logits. Clustering/probe fitted ONLY on training representations.

    generated: independent marginal samples (training aggregated posterior for SFA).
    posterior_pred: p(x'|z), z~q(z|held-out x). shuffled_pred uses same x0, permuted z.
    """
    out = {'n_eval': len(true_x), 'vendi_kernel': 'cosine_unit_diagonal',
           'cluster_protocol': 'kmeans_fit_train_predict_heldout',
           'representation': 'posterior_mean_or_deterministic_encoder'}
    image_data = dataset in ('mnist', 'emnist', 'pendulum')
    tx = pixels(true_x) if image_data else np.asarray(true_x)
    if generated is not None:
        gx = pixels(generated) if image_data else np.asarray(generated)
        out['vendi_marginal_cosine'] = vendi_cosine(gx)
        if dataset == 'pinwheel':
            out['w1_marginal_euclidean_raw'] = w1_euclidean(tx, gx)
    if posterior_pred is not None:
        pp = pixels(posterior_pred) if image_data else np.asarray(posterior_pred)
        out['posterior_predictive_rmse'] = float(np.sqrt(np.mean((tx-pp)**2)))
        out['vendi_posterior_predictive_cosine'] = vendi_cosine(pp)
        if dataset in ('mnist', 'emnist'):
            out['posterior_predictive_ssim'] = paired_ssim(tx, pp)
        if shuffled_pred is not None:
            sp = pixels(shuffled_pred) if image_data else np.asarray(shuffled_pred)
            out['matched_noise_latent_change_mse'] = float(np.mean((pp-sp)**2))
            out['shuffled_rmse'] = float(np.sqrt(np.mean((tx-sp)**2)))
            out['shuffle_mse_gap'] = float(np.mean((tx-sp)**2)-np.mean((tx-pp)**2))
            if dataset in ('mnist', 'emnist'):
                out['shuffled_ssim'] = paired_ssim(tx, sp)
                out['shuffle_ssim_gap'] = out['posterior_predictive_ssim']-out['shuffled_ssim']
    if train_z is not None and test_z is not None and train_y is not None and test_y is not None:
        train_z, test_z = np.asarray(train_z), np.asarray(test_z)
        train_y, test_y = np.asarray(train_y), np.asarray(test_y)
        if train_y.ndim == 1 and test_y.ndim == 1:
            k = int(n_clusters or len(np.unique(train_y)))
            scaler = StandardScaler().fit(train_z)
            a, b = scaler.transform(train_z), scaler.transform(test_z)
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(a)
            pred = km.predict(b)
            out['nmi_trainfit'] = float(normalized_mutual_info_score(test_y, pred))
            out['ari_trainfit'] = float(adjusted_rand_score(test_y, pred))
            # For comparison with the paper's transductive clustering protocol, named separately.
            pred_testfit = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(b)
            out['nmi_testfit_paper_style'] = float(normalized_mutual_info_score(test_y, pred_testfit))
            out['ari_testfit_paper_style'] = float(adjusted_rand_score(test_y, pred_testfit))
            if len(np.unique(train_y)) > 1:
                probe = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(a, train_y)
                out['linear_probe_accuracy'] = float(accuracy_score(test_y, probe.predict(b)))
            if dataset == 'pinwheel':
                # Decoder angular utility: train-only linear map, no test fitting.
                out['cluster_count'] = k
    return out
