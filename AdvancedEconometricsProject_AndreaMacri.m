% We show some (interesting) properties of the Maximum Likelihood
% Estimator.
% The topic is the example of regression with normal errors (slides 258 -260 and 266-268).
% Authors: Andrea Macri - Lieven Govaerts

clear
clf
options=optimoptions('fminunc','Algorithm','quasi-newton','OptimalityTolerance',10^(-6),'Display','off');

% Sample sizes for the simulation, adapt this for performance vs accuracy:
n=10^4;

% Sample sizes for the asymptotic normality of MLE
R = 10^4;
norm_n = 250; % For performance reasons, pick 250 samples here, from the MLE
% plots we see that the estimation of the beta's is stable
% as of 250 samples.


% Generate a dataset of normally distributed samples.
X=[ones(n,1),randn(n,1)];
beta=[1;2];
epsilon=randn(n,1);
y=X*beta+epsilon;

% Transform the response variable of the first dataset by g(x) = x^3
tbeta=beta.^3;
ty=X*tbeta+epsilon;

% Initialize the working variables
ols=zeros(n,2); ssr=zeros(n,1);
ml_b=zeros(n,2); ml_var=zeros(n,2);
retml=zeros(n,2);
tml=zeros(n,2);

% Run n simulations, each with an incrementing # of samples.
for j=1:n
    x=X(1:j,:); h=y(1:j); th=ty(1:j);
    
    % OLS estimates
    ols(j,:)=(x\h)';
    res=y-X*ols(j);
    ssr=res'*res;
    ssqu=ssr/n; % OLS variance. TODO: we don't do anything with this now. Plot?
    
    % MLE estimates
    start=[3,3]; %starting values
    [ml_b(j,:),fval,exitflag,output,grad,hessian]=fminunc(@(b) -loglik(h,x,b(1:2)),start,options); %to find the hessian automatically
    ml_info_mat=hessian^-1; %the variance of the estimator is low since it converges to true value rapidly
    ml_var=sum(trace((hessian)^-1)); %asy variance

    % Equivariance under reparametrization of the MLE
    tml(j,:)=fminunc(@ (bb) -loglik(th,x,bb(1:2)),start,options);
    retml(j,:)=nthroot(tml(j,:),3);
end


diff(j,:)=retml(j,:)-ml_b(j,:);
diff(j,:);


% Asymptotic normality of the MLE:
% sqrt(n) * (theta-hat - theta-0) ->d normal distribution (slide 261)
% Simulate by creating R samples each of size norm_n, estimate theta for
% each sample. The (theta - theta 0) distribution should be bivariate normal.
fisher_db = zeros(R,2);
for i = 1:R
    % Generate R datasets of norm_n normally distributed samples.
    x=[ones(norm_n,1),randn(norm_n,1)];
    epsilon=randn(norm_n,1);
    y = x * beta + epsilon;
    
    start = [3,3];
    [b,fval,exitflag,output,grad,hessian] = fminunc(@(b) -loglik(y,x,b(1:2)),start,options);
    % Calculate the difference between true value and estimation
    fisher_db(i,:) = b' - beta;
end




figure(1)
% first plot is the mle estimation
subplot(2,2,1), plot(ml_b,'DisplayName','ml estimate of beta0 and of beta1'), xlim([0,500]),ylim([0,3]), title('ML estimate of beta0 (exp. 1) and of beta1 (exp. 2)');
% second plot plots the ols estimation
subplot(2,2,2), plot(ols,'DisplayName','ols estimate of beta0 and of beta1'),xlim([0,500]),ylim([0,3]),title('OLS estimate of beta0 (exp. 1) and of beta1 (exp. 2)');
% third plot plots the transformed pmt mle
subplot(2,2,3), plot(tml,'DisplayName','transformed ml estimate of beta0 and of beta1'), xlim([0,500]), title('Transformed ML estimate of beta0 (exp. 1) and of beta1 (exp. 8)');
% last plot plots the retransformed mle
subplot(2,2,4), plot(retml,'DisplayName','retml estimate of beta0 and of beta1'), xlim([0,500]), title('Retransformed estimate of beta0 (exp. 1) and of beta1 (exp. 2)');

figure(2)
subplot(1,3,1), hist3(fisher_db, {-1:.1:1 -1:.1:1}, 'CDataMode','auto','FaceColor','interp');
subplot(1,3,2), histfit(fisher_db(:,1),50,'normal');%fitted normal to the histogram, it's a bit
subplot(1,3,3), histfit(fisher_db(:,2),50,'normal');%tall since the variance is very low


function logL=loglik(y,X,b)
    logL=sum(log(normpdf(y-X*b')));
end