function [S] = m_laplacian_score(M, f)
% M: transition probabilities of random walk of set time (n x n array)
% f: signal (as 1 x n array)
% 
% Note: if communities were computed using generalizedLouvain, M at time t
% can be computed as M = lovain_FNL(admat, t, P)/diag(P.pi), where P are
% the parameters returned by partition_stability.
%
% https://github.com/michaelschaub/generalizedLouvain
%

D = diag(sum(M.'));

f = graph_mean(D, f);
S = zeros(size(f, 1), 1);

if sum(M, 2) ~= ones(size(M, 1), 1)
    M = M / diag(sum(M, 2));
end

M = eye(size(M, 1)) - M.';

for i = 1:size(f, 1)
    S(i) = f(i, :) * D * M * transpose(f(i, :)) / (f(i, :) * D * transpose(f(i, :)));
end
end


function [f] = graph_mean(D, f)
norm = sum(sum(D));
for i = 1:size(f, 1)
    f(i, :) = f(i, :) - sum(f(i, :) * D) / norm;
end
end
