function [X,D] = KR_DL(A,b_analytic,Nodes)
k = 15;  % 设置稀疏度
numnodes = size(Nodes,1);

D = A;
for ii = 1:size(A,2)
    D(:,ii)=D(:,ii)/norm(D(:,ii));
end

% 给一些变量和参数赋初值
Y = b_analytic;
X = zeros(numnodes, 1);
nIter = 10;  % 设置迭代次数
s = 68; 
G = ceil(numnodes/s);  % 子字典的数量

for i = 1:nIter

    err0 = norm(A*X-b_analytic);

    %% Step1.Sparse Coding，在稀疏编码阶段，采用K-LIMAPS算法求解稀疏系数向量X
    X = klimaps(D,pinv(D),Y,k,10);

    %% Step2.Dictionary Update，先对字典D进行分组，然后使用正交Procrustes分析和奇异值分解更新字典D
    Pop = sum(abs(X),2); 
    [~, M] = sort(Pop,'ascend');
    sizeM = numel(M);
    
    blockSizes = [ones(1,G-1)*s, sizeM-(G-1)*s];
    blockEnds = cumsum(blockSizes);
    blockStarts = [0 blockEnds(1:end-1)] + 1;
    
    XYt = X*Y';
    XXt = X*X'; 
    
    % 更新每个子字典，所有子字典更新完毕即完成对整个字典的更新
    for g = 1:G
        I = M(blockStarts(g):blockEnds(g));
        IC = true(1,numnodes);
        IC(I) = false;
        I = not(IC);
        
        % 通过奇异值分解求解正交Procrustes问题
        HEt = D(:,I)*(XYt(I,:)-XXt(I,IC)*D(:,IC)'); 
        [U,~,V] = rsvd(HEt,sum(I)); 
        % 更新子字典
        D(:,I) = V*U'*D(:,I); 
    end
    err1 = norm(A*X-b_analytic);
    if (abs(err1-err0)<1e-6)
        break;
    end
end
end

% random SVD
function [U,S,V] = rsvd(A,K)
[M,N] = size(A);
P = min(2*K,N);
X = randn(N,P);
Y = A*X;
W1 = orth(Y);
B = W1'*A;
[W2,~,V] = svd(B,'econ');
U = W1*W2;
K=min(K,size(U,2));
U = U(:,1:K);
%S = S(1:K,1:K);
S = [];
V=V(:,1:K);
end


function Alpha = klimaps(D,Dinv,X,K,max_iter)
% K-LIMAPS 算法
%       min  |X - D*ALPHA|_F     s.t.  |Alpha|_0 <= K
%      ALPHA
% D是归一化后的字典，Dinv表示其伪逆，X是已知的观测数据/信号，K是稀疏度，max_iter表示迭代次数 
Alpha = Dinv*X;
m  = size(D,2);
N = size(Alpha,2);
P = eye(m) - Dinv*D;
a = sort(abs(Alpha),'descend');
Lambda = repmat(1./a(K+1,:),m,1);
%Lambda = 1.01*ones(1,N);
for i = 1:max_iter
    %Beta = Alpha.*(1-exp(-Lambda.*abs(Alpha)));
    %Alpha = Beta-Dinv*(D*Beta-X);
    Alpha = Alpha - P*(Alpha.*exp(-Lambda.*abs(Alpha)));
    [a,idx] = sort(abs(Alpha),'descend');
    Lambda = repmat(1./a(K+1,:),m,1);
end
for i = 1:N
    Alpha(idx(K+1:m,i),i) = 0;  
    Alpha(idx(1:K,i),i) = pinv(D(:,idx(1:K,i)))*X(:,i); 
end
Alpha = sparse(Alpha);
end

