%%-----------------------HOMEWORK-2-TEXTURE------------------------------%%
%                                                                         %
%-------------------------------------------------------------------------%
clear all
close all
% Add some directories to the path: functions
addpath('functions/');
addpath('Texture/');
%--------------------------------------------------------------------------
%% Step 1: Loading all images from Texture files
% nxn is size of each image
n=256;
% M_I is matrix which includes grayscale matrix of each image
M_I=[];
% num is number of images in Cartoon
num=8;
% Find all grayscale matrix and then add all into M_I
for i=1:8
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
% Sigma is noise level;
sigma=0.1;

% K is name of K-th image
K=8;

% M0 is grayscale matrix of K-th image 
M0=M_I(1:n,(K-1)*n+1:K*n);
% M is noised grayscale matrix of K-th image
M=M0+sigma*randn(n);
% M is noised grayscale matrix of all images
%-% M=M_I+sigma*randn(n,size(M_I,2))
% Plot all images and all noise images
figure;
subplot(1,2,1);imshow(M0);title('Original image');
subplot(1,2,2);imshow(M);title('Noising images');
%-% subplot(2,1,1);imshow(M_I);title('All original Cartoon images');
%-% subplot(2,1,2);imshow(M);title('All noising images');
%--------------------------------------------------------------------------%

%% Step 2: Patch Extraction
% Size of patches: w
w=12;
% Number of patches: m
m=40*w^2;
% Random patch location
x=floor(rand(1,1,m)*(n-w))+1;
y=floor(rand(1,1,m)*(n-w))+1;
% Extract lots of patches
[dY,dX]=meshgrid(0:(w-1),0:(w-1));
Xp=repmat(dX,[1 1 m])+repmat(x,[w w 1]);
Yp=repmat(dY,[1 1 m])+repmat(y,[w w 1]);
P=M(Xp+(Yp-1)*n);
% "Remove mean, since we are going to learn a dictionary of zero-mean and
% unit norm atom. The mean of the patches is close to being noise free
% anyway"
P=P-repmat(mean(mean(P)),[w w]);
% Reshape so that each P(:,i) corresponds to a patch
P=reshape(P,[w^2 m]);
% Display a few random patches
%-% plot_dictionary(P,[],[8 12]);
%--------------------------------------------------------------------------

%% Step 3: Sparse Coding
% Number of atoms in the dictionary: p
p=w^2;
% The initial dictionary is computed by a random selection of patches.
sel=randperm(m);
sel=sel(1:p);
D=P(:,sel);
% Normalize the atoms
D=D./repmat(sqrt(sum(D.^2)),[w^2, 1]);
% The sparse coding is obtained by minimizing a L1 penalized optimization.
% The value of lambda controls the sparsity of the coeficients. Since L1
% regularization is similar to soft threshoding, we use the usual
% 3/2*sigma value
lambda=1.5*sigma;
% The gradient descent step size mu is related to the operator norm of the
% dictionary.
mu=1.9/norm(D)^2;
% Initialize the coefficients.
X=zeros(p,m);
% One step of iteration, for all the patches together
X=perform_thresholding(X+mu*D'*(P-D*X),lambda*mu);
%--------------------------------------------------------------------------

%% Step 4: Automatic Set of the lambda Value
% The value of lambda was chosen arbitrary, and was the same for all the
% patches.. Here we use an independent value of lambda (i) for each patch
% P(:,i)
lambda=zeros(m,1)+1.5*sigma;
% We let lambda(i) evolves during the optimization so that one has 
% norm(P(:,i)-D*X(:,i),'fro')=rho*w*sigma, where rho is a damping factor
% close to 1. The value w*sigma is approximately the amount of noise that
% contaminate P(:,i);
rho=1.4;
error_target=rho*w*sigma;
% Using an idea originaly developed by Antonin Chambolle, we use the
% following update rule during the iterative thresholding.
lambda=lambda*error_target./sqrt(sum((P-D*X).^2))';
%--------------------------------------------------------------------------


%% Step 5: Update the dictionary 
% Updating the dictionary is achieve by minimizing norm(D*X-P,'fro') over
% all posible D. The solution by a pseudo-inverse.
D=P*pinv(X);
% The atoms are then normalized
D=D./repmat(sqrt(sum(D.^2)),[w^2,1]);
%-% plot_dictionary(D,X,[8 12])
%--------------------------------------------------------------------------


%% Step 6: Denoising by Sparse Coding
% The denoising of the image is obtained by sparse coding a large
% collection of patches (idelly all the patches).

% Overlap parameter (q=w is implies no overlap)
q=4;
% Regularly space positions for the extraction of patches
[y,x]=meshgrid(1:q:n-w/2,1:q:n-w/2);
m=size(x(:),1);
Xp=repmat(dX,[1 1 m])+repmat(reshape(x(:),[1 1 m]),[w w 1]);
Yp=repmat(dY,[1 1 m])+repmat(reshape(y(:),[1 1 m]),[w w 1]);
% Ensure boundary conditions
Xp(Xp>n)=2*n-Xp(Xp>n);
Yp(Yp>n)=2*n-Yp(Yp>n);
% Extract a large sub-set of regularly sampled patches.
P=M(Xp+(Yp-1)*n);
P=reshape(P,[w^2, m]);
% Save the mean of patches appart, and remove it.
a=mean(P);
P=P-repmat(a,[w^2 1]);
% Set a target error for denoising
rho=1;
error_target=rho*w*sigma;
% Update the dictionary again
sel=randperm(m);
sel=sel(1:p);
D=P(:,sel);
X=zeros(p,m);
X=perform_thresholding(X+mu*D'*(P-D*X),lambda*mu);
D=P*pinv(X);
D=D./repmat(sqrt(sum(D.^2)),[w^2,1]);
% Approximated patches
PA=reshape(D*X,[w w m]);
% Insert back the mean
PA=PA-repmat(mean(mean(PA)),[w w]);
PA=PA+reshape(repmat(a,[w^2 1]),[w w m]);
% To obtain the denoising, we average the value of the approximated patches
% PA that overlap
W=zeros(n,n);
M1=zeros(n,n);
for i=1:m
    x=Xp(:,:,i);
    y=Yp(:,:,i);
    M1(x+(y-1)*n)=M1(x+(y-1)*n)+PA(:,:,i);
    W(x+(y-1)*n)=W(x+(y-1)*n)+1;
end
M1=M1./W;
% Display the result
figure;
hold on
subplot(2,1,1);imshow(M1);title('Denoised');
subplot(2,1,2);imshow(M);title('Noising');
