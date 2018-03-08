
x=[5; 3; 2];
d= size(x,1);
mu=[1;2;2];
sigma=[1 0 0; 0 5 2; 0 2 5];
y = mahalonobis(x,mu,sigma)
g = discriminant(x,mu,sigma,0.5);
m = generateNormalRandomNumbers(mu,sigma,100);
d = getEuclidianDistance([0 0 0], [0 1 2]);
data = [-5.01 -8.12 -3.68 1;
       -5.43 -3.48 -3.54 1;
       1.08 -5.52 1.66 1;
       0.86 -3.78 -4.11 1;
       -2.67 0.63 7.39 1;
       4.94 3.29 2.08 1;
       -2.51 2.09 -2.59 1;
       -2.25 -2.13 -6.94 1;
       5.56 2.86 -2.26 1;
       1.03 -3.33 4.33 1;
       -0.91 -0.18 -.05 2;
       1.30 -2.06 -3.53 2;
       -7.75 -4.54 -.95 2;
       -5.47 0.50 3.92 2;
       6.14 5.72 -4.85 2;
       3.60 1.26 4.36 2;
       5.37 -4.63 -3.65 2;
       7.18 1.46 -6.66 2;
       -7.39 1.17 6.30 2;
       -7.50 -6.32 -.31 2;
       5.35 2.26 8.13 3;
       5.12 3.22 -2.66 3;
       -1.34 -5.31 -9.87 3;
       4.48 3.42 5.19 3;
       7.11 2.39 9.21 3;
       7.17 4.33 -0.98 3;
       5.75 3.97 6.65 3;
       0.77 0.27 2.41 3;
       0.90 -0.43 -8.71 3;
       3.52 -0.36 6.43 3];
   [n,m] = size(data);
    means = mean(data(:,1:3))
   %sigmas = cov(data(:,1:3));
   %y = mahalonobis(x,means', sigmas)
   class1_data=data(1:10,:);
   class2_data=data(11:20,:);
   class3_data=data(21:30,:);
   m1=mean(class1_data);
   m2=mean(class2_data);
   m3=mean(class3_data);
   means = vertcat(m1,m2,m3);
   means = means(:,1:3);
   s1= cov(class1_data(:,1:3));
   s2= cov(class2_data(:,1:3));
   s3= cov(class3_data(:,1:3));
   sigmas = vertcat(s1,s2,s3);
   priors=[.5 .5 0];
  count =0;
  for i=1:n
    x=data(i:i,1:3);
    class= classify(x', means, sigmas,priors,3);
    fprintf('classified as: %d , original %d\n',class,data(i:i,4:4));
    if class == data(i:i,4:4) 
        count=count+1;
    end
  end
  fprintf("total correct %d\n",count);
  

function g = discriminant(p,mu,sigma,prior)

  f = mahalonobis(p,mu,sigma);
  d= size(p,1);
  g = -0.5*f - d/2*log(2*pi) -(1/2)*log(det(sigma)) + log(prior);
end

function f= mahalonobis(p,mu,sigma)
   
  x= p - mu;
  y= x';
  f = y*inv(sigma)*x;
 
end
function m = generateNormalRandomNumbers(mu,sigma,size)
% this function generates size number of multivariate normal random numbers
%mu is the mean. 
%sigma is the covariance matrix
%size is the number of samples to be generated
    m = mvnrnd(mu,sigma,size);
end
function d = getEuclidianDistance(point1, point2)
% this function measures the euclidian distance between two points
% point1 is the first point
%point2 is the 2nd point
x= vertcat(point1,point2);
  d = pdist(x,'euclidean');
end
function class = classify(point,means,sigmas,priors,features)
    point=point';
    point = point(:,1:features);
    point=point';
    %means=means(1:,2:2)
    max= -100000;
    class=0;   
    for i = 1:3
        s=sigmas((i-1)*3+1:(i-1)*3+features,1:features);
        m=means(i:i,1:features);
        m=m';
        x= mahalonobis(point,m,s);
      d = discriminant(point,m,s,priors(:,i));
      if d >max
          max=d;
          class=i;
      end
    end
        
   
end