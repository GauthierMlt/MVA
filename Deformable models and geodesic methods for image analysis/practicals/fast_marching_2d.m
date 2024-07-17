warning off
addpath('toolbox_signal')
addpath('toolbox_general')
addpath('toolbox_graph')
% addpath('solutions/fastmarching_1_2d')
warning on


% We load the input image f.
clear options;
n = 300;
name = 'road2';
f = rescale( load_image(name, n) );

% Display the image.

clf;
imageplot(f);

% Define start and end points x0 and x1 (note that you can use your own points).

x0 = [14;161];
x1 = [293;148];

% The metric is defined according to f
% in order to be low at pixel whose value is close to f(x). A typical example is
% W(x)=?+|f(x0)?f(x)|
% where the value of ?>0 should be increased in order to obtain smoother paths.


epsilon = 1e-2;
W = epsilon + abs(f-f(x0(1),x0(2)));

% Display the metric W

clf;
imageplot(W);

% Set options for the propagation: infinite number of iterations, and stop when the front hits the end point.

options.nb_iter_max = Inf;
options.end_points = x1;

% Perform the propagation, so that D(a,b) is the geodesic distance between the pixel x1=(a,b) and the starting point x0. Note that the function |perform_fast_marching| takes as input the inverse of the metric 1/W(x).

[D,S] = perform_fast_marching(1./W, x0, options);



% Display the propagated distance map D
% . We display in color the distance map in areas where the front has propagated, and leave in black and white the area where the front did not propagate.

clf;
hold on;
imageplot( convert_distance_color(D,f) );
h = plot(x0(2),x0(1), '.r'); set(h, 'MarkerSize', 25);
h = plot(x1(2),x1(1), '.b'); set(h, 'MarkerSize', 25);


%% Exercice 1

% Using |options.nb_iter_max|, display the progressive propagation. This corresponds to displaying the front {x,D(x)?t} for various arrival times t.
% You can use the function "convert_distance_fun(D,f)"


%% Geodesic curve extraction

% Once the geodesic distance map D(x) to a starting point x0 is computed, the geodesic curve between any point x1 and x0 extracted through gradient descent
% ??(t)=??t?D(?(t)),
% where ?t>0 controls the parameterization speed of the resulting curve. To obtain unit speed parameterization, one can use ?t=??D(?(t))??1
% 
% .
% 
% Recompute the geodesic distance map D
% on the whole grid.

options.nb_iter_max = Inf;
options.end_points = [];
[D,S] = perform_fast_marching(1./W, x0, options);

% Display D

clf;
imageplot(D);
colormap jet(256);

% Compute the gradient G0(x)=?D(x)?R2 of the distance map. Use centered differences.

options.order = 2;
G0 = grad(D, options);

% Normalize the gradient to obtained G(x)=G0(x)/?G0(x)?, in order to have unit speed geodesic curve (parameterized by arc length).

G = G0 ./ repmat( sqrt( sum(G0.^2, 3) ), [1 1 2]);

clf;
imageplot(G);
colormap jet(256);


% The geodesic is then numerically computed using a discretized gradient descent, which defines a discret curve (?k)k using
% ?k+1=?k??G(?k)
% where ?k?R2 is an approximation of ?(t) at time t=k?, and the step size ?>0
% 
% should be small enough.
% 
% Step size ?
% for the gradient descent.

tau = .8;

% Initialize the path with the ending point.

gamma = x1;

% Define a shortcut to interpolate G
% at a 2-D points. Warning: the |interp2| switches the role of the axis ...



Geval = @(G,x)[interp2(1:n,1:n,G(:,:,1),x(2),x(1)); ...
             interp2(1:n,1:n,G(:,:,2),x(2),x(1)) ];
         
% Compute the gradient at the last point in the path, using interpolation.

g = Geval(G, gamma(:,end));

% Perform the descent and add the new point to the path.

gamma(:,end+1) = gamma(:,end) - tau*g;

%% Exercice 2
% Perform the full geodesic path extraction by iterating the gradient descent. 
% You must be very careful when the path become close to x0, because the distance function is not differentiable at this point. You must stop the iteration when the path is close to x0.







% Display the curve on the image background.

clf; hold on;
imageplot(f);
h = plot(gamma(2,:),gamma(1,:), '.b'); set(h, 'LineWidth', 2);
h = plot(x0(2),x0(1), '.r'); set(h, 'MarkerSize', 25);
h = plot(x1(2),x1(1), '.b'); set(h, 'MarkerSize', 25);
axis ij;

% Display the curve on the distance background.

clf; hold on;
imageplot(D); colormap jet(256);
h = plot(gamma(2,:),gamma(1,:), '.b'); set(h, 'LineWidth', 2);
h = plot(x0(2),x0(1), '.r'); set(h, 'MarkerSize', 25);
h = plot(x1(2),x1(1), '.b'); set(h, 'MarkerSize', 25);
axis ij;

%% Exercice 3

% Study the influence of the ? parameter.

%% Exercice 4
% Perform the shortest path extraction for various images such as 'cavern' or 'mountain'. oad radient isplay

f = load_image('cavern',n);
x0 = [45;280]; x1 = [275;25];





%% Edge-based Geodesic Methods

% It is possible to extract the boundary of an object using shortest paths that follows region of high gradient.
% 
% First we load an image f

n = 256;
name = 'cortex';
f = rescale( sum(load_image(name,n),3) );

% Display it.

clf;
imageplot(f);




% An edge-attracting potential W(x)
% should be small in regions of high gradient. A popular choice is
% W(x)=1?+G??G(x)$whereG(x)=??f(x)?,
% and where G? is a Gaussian kernel of variance ?2
% Compute the gradient norm G(x)

G = grad(f,options);
G = sqrt( sum(G.^2,3) );

% Smooth it by G?.

sigma = 3;
Gh = perform_blurring(G,sigma);

% Display the smoothed gradient G?G?

clf;
imageplot(Gh);

% Compute the metric.


epsilon = 0.01;
W = 1./( epsilon + Gh );

clf;
imageplot(W);

% Set two starting point S={x10,x20} (you can use other points).

x0 = [ [136;53] [123;205]];

% Compute the Fast Marching from these two base points.
options.nb_iter_max = Inf;
options.end_points = [];
[D,S,Q] = perform_fast_marching(1./W, x0, options);

% Display the geodesic distance (with color normalization).
clf; hold on;
imageplot( perform_hist_eq(D,'linear') );
h = plot(x0(2,:),x0(1,:), '.r'); set(h, 'MarkerSize', 25);
colormap jet(256);

% The Voronoi segmentation associated to S is
% Ci={x,?j?i,d(xi0,x)?d(xj0,x)}.
% This Voronoi segmentation is computed during the Fast Marching propagation and is encoded in the partition function Q(x)
% using Ci={x,Q(x)=i}
% Display the distance and the Voronoi segmentation.

clf; hold on;
A = zeros(n,n,3); A(:,:,1) = rescale(Q); A(:,:,3) = f;
imageplot(A);
h = plot(x0(2,:),x0(1,:), '.g'); set(h, 'MarkerSize', 25);


%% Exercice 5

% Extract the set of points that are along the boundary of the Voronoi region. 
% This corresponds for instance to the points of the region {x,Q(x)=1} that have one neighbor inside the region {x,Q(x)=2}. Compute the geodesic distance D(x) at these points, and choose two points a and b on this boundary that have small values of D. 
% hint: you can use a convolution |U=conv2(double(Q==2),h,'same')| with a ell chose kernel |h| to located the points |U>0| with at least 1 eighbor.

%% Exercice 6
% Extract the geodesics joining a and b to the two starting points (this makes 4 geodesic curves). Use them to perform segmentation. D1 = D; D1(D1==Inf) = max(D1(D1~=Inf)); isplay the curves


%% Vessel Segmentation and Centerline Extraction
% One can extract a network of geodesic curve starting from a central point to detect vessels in medical images.
% Load an image. This image is extracted from the <http://www.isi.uu.nl/Research/Databases/DRIVE/ DRIVE database> of retinal vessels.

n = 256;
name = 'vessels';
f = rescale(load_image(name, n));

clf;
imageplot(f);



% We clean the image by substracting the smoothly varying background
% f1=f?G??f,
% where G? is a Gaussian kernel of variance ?2. Computing f1 corresponds to a high pass filtering.

sigma = 20;
f1 = perform_blurring(f,sigma) - f;

% Display this normalized image.

clf;
imageplot(f1);

% We compute a metric tthat is small for large values of f1:
% W(x)=?+|f1(x)?c|$wherec=maxxf1(x).

c = max(f1(:));
epsilon = 1e-2;
W = epsilon + abs(f1-c);

clf,
imageplot(W);

% Select a central point x0 for the network.

x0 = [142;226];

%% Exercice 7
% Perform partial propagations from x0

x1 = [ [175;5] [21;42] [48;133] [244;78] [191;40] ...
         [100;13] [66;42] [183;66] [220;117]];

%% Exercice 8
% Extract geodesics joining several points x1 to the central point x0. radient xtract centerlines isplay the curves

