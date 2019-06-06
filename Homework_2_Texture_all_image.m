%%--------------HOMEWORK-2-CARTOON-ALL IMAGES----------------------------%%
%                                                                         %
%-------------------------------------------------------------------------%
clear all
close all
addpath('functions/');
addpath('Texture/');
n=256;
M_I=[];
num=8;
for i=1:num
    if (i==1)|(i==4)
        M0=imread(sprintf('%i.jpg',i));
        M0=imresize(M0,[n n]);
        M0=im2double(M0);
        M_I=[M_I M0];
    else
        M0=imread(sprintf('%i.jpg',i));
        M0=rgb2gray(M0);
        M0=imresize(M0,[n n]);
        M0=im2double(M0);
        M_I=[M_I M0];
    end
end
sigma=0.18;
M_A=M_I+sigma*randn(n,size(M_I,2));
%-%M_A=M_I+sigma*randn(n,size(M_I,2));
M_N=[];
for K=1:num
    M0=M_I(1:n,(K-1)*n+1:K*n);
    M=M0+sigma*randn(n);
    w=12;
    m=40*w^2;
	x=floor(rand(1,1,m)*(n-w))+1;
    y=floor(rand(1,1,m)*(n-w))+1;
    [dY,dX]=meshgrid(0:(w-1),0:(w-1));
    Xp=repmat(dX,[1 1 m])+repmat(x,[w w 1]);
    Yp=repmat(dY,[1 1 m])+repmat(y,[w w 1]);
    P=M(Xp+(Yp-1)*n);
    P=P-repmat(mean(mean(P)),[w w]);
    P=reshape(P,[w^2 m]);
    p=w^2;
    sel=randperm(m);
    sel=sel(1:p);
    D=P(:,sel);
    D=D./repmat(sqrt(sum(D.^2)),[w^2, 1]);
    lambda=1.5*sigma;
    mu=1.9/norm(D)^2;
    X=zeros(p,m);
    X=perform_thresholding(X+mu*D'*(P-D*X),lambda*mu);
    lambda=zeros(m,1)+1.5*sigma;
    rho=1.4;
    error_target=rho*w*sigma;
    lambda=lambda*error_target./sqrt(sum((P-D*X).^2))';
    D=P*pinv(X);
    D=D./repmat(sqrt(sum(D.^2)),[w^2,1]);
    q=4;
    [y,x]=meshgrid(1:q:n-w/2,1:q:n-w/2);
    m=size(x(:),1);
    Xp=repmat(dX,[1 1 m])+repmat(reshape(x(:),[1 1 m]),[w w 1]);
    Yp=repmat(dY,[1 1 m])+repmat(reshape(y(:),[1 1 m]),[w w 1]);
    Xp(Xp>n)=2*n-Xp(Xp>n);
    Yp(Yp>n)=2*n-Yp(Yp>n);
    P=M(Xp+(Yp-1)*n);
    P=reshape(P,[w^2, m]);
    a=mean(P);
    P=P-repmat(a,[w^2 1]);
    rho=1;
    error_target=rho*w*sigma;
    sel=randperm(m);
    sel=sel(1:p);
    D=P(:,sel);
    X=zeros(p,m);
    X=perform_thresholding(X+mu*D'*(P-D*X),lambda*mu);
    D=P*pinv(X);
    D=D./repmat(sqrt(sum(D.^2)),[w^2,1]);
    PA=reshape(D*X,[w w m]);
    PA=PA-repmat(mean(mean(PA)),[w w]);
    PA=PA+reshape(repmat(a,[w^2 1]),[w w m]);
    W=zeros(n,n);
    M1=zeros(n,n);
    for i=1:m
        x=Xp(:,:,i);
        y=Yp(:,:,i);
        M1(x+(y-1)*n)=M1(x+(y-1)*n)+PA(:,:,i);
        W(x+(y-1)*n)=W(x+(y-1)*n)+1;
    end
    M1=M1./W;
    M_N=[M_N M1];
end
figure;
hold on
subplot(3,1,1);imshow(M_I);title('Original');
subplot(3,1,2);imshow(M_A);title('Noising');
subplot(3,1,3);imshow(M_N);title('Denoised');
%-%subplot(1,3,1);imshow(M_I);title('Original');
%-%subplot(1,3,2);imshow(M_A);title('Noising');
%-%subplot(1,3,3);imshow(M_N);title('Denoised');
