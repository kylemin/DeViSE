function [cost, grad] = deviseLoss(w, t_random, t_label, labels, v, mgin)
m = size(v, 2);
labelDim = size(t_random, 1);

tmp = w * v;
hl = diag(t_label * tmp)';
hr = t_random * tmp;
hingeM = - bsxfun(@minus, hr, hl);
labelM = hingeM < mgin & ~full(ind2vec(labels', labelDim));
t = bsxfun(@times, t_label, sum(labelM)');
t = t - labelM' * t_random;
grad = -(v*t)';

hingeM = max(0, mgin - hingeM);
cost = sum(sum(hingeM))/m - mgin;
grad = grad/m;
