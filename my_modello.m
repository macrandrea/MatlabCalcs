function [y1,y2]=My_Prob(S0,K,sigma0,T,r,b,eta,Lambda,rho,phi)
x = log(S0);
lambda=(2*b)/sigma0^2;
delta=(lambda-(2*eta-1))/(1-eta);
P1=(-phi^2-i*phi)/2;
P2=(-phi^2+i*phi)/2;
Q1=(2*sqrt(sigma0)*(i*phi-rho));
Q2=i*phi*2*sqrt(sigma0);
R=2;
d1=sqrt(Q1-4*P1*R);
g1=(Q1-d1)/(Q1+d1);
b1=rho*2*sqrt(sigma0);
d2=sqrt(Q2-4*P2*R);
g2=(Q1-d2)/(Q2+d2);
b2=0;
%Riccatis
D1=((b1-Q1+d1)/4)*((1-exp(d1*T))/(1-g1*exp(d1*T)));
C1=r*i*phi*T+delta/2*(((b1-Q1+d1)/4)*T-2*log((1-exp(d1*T))/(1-g1*exp(d1*T))));
D2=((Q2+d2)/4)*((1-exp(d2*T))/(1-g2*exp(d2*T)));
C2=r*i*phi*T+delta/2*(((Q2+d2)/4)*T-2*log((1-exp(d2*T))/(1-g2*exp(d2*T))));
%%%%%
f1 = exp(C1 + D1*sigma0 + i*phi*x);
y1 =  real(exp(-i*phi*log(K))*f1/(i*phi)); %no more @(phi) 
% q1 = integral(y1,0.1,10000,'RelTol',0,'ArrayValued',true);
%%%%%
f2=exp(C2 + D2*sigma0 + i*phi*x);
y2=real(exp(-i*phi*log(K))*f2/(i*phi));
end
