warning off
addpath('toolbox_signal')
addpath('toolbox_general')
addpath('toolbox_graph')
addpath('solutions/fastmarching_0_implementing')
warning on

%% Navigating on the grid
% We use a cartesian grid of size n×n, and defines operators to navigate in the grid.
% We use a singe index i?{1,…,n2}
% to index a position on the 2-D grid.
% Size of the grid.

n = 40;

neigh = [[1;0] [-1;0] [0;1] [0;-1]];

boundary = @(x)mod(x-1,n)+1;

ind2sub1 = @(k)[rem(k-1, n)+1; (k - rem(k-1, n) - 1)/n + 1]; 
sub2ind1 = @(u)(u(2)-1)*n+u(1);
Neigh = @(k,i)sub2ind1( boundary(ind2sub1(k)+neigh(:,i)) );



% For simplicity of implementation, we use periodic boundary conditions.

boundary = @(x)mod(x-1,n)+1;



% For a given grid index |k|, and a given neighboring index k in 1,2,3,4
% , |Neigh(k,i)| gives the corresponding grid neighboring index.

ind2sub1 = @(k)[rem(k-1, n)+1; (k - rem(k-1, n) - 1)/n + 1]; 
sub2ind1 = @(u)(u(2)-1)*n+u(1);
Neigh = @(k,i)sub2ind1( boundary(ind2sub1(k)+neigh(:,i)) );




%% Dijkstra Algotihm

% The Dijkstra algorithm compute the geodesic distance on a graph. We use here a graph whose nodes are the pixels, and whose edges defines the usual 4-connectity relationship.
% 
% In the following, we use the notation i?j
% to indicate that an index j is a neighbor of i
% 
% on the graph defined by the discrete grid.
% 
% The metric W(x)
% . We use here a constant metric.

W = ones(n);

% Set S={x0} of initial points.

x0 = [n/2;n/2];

% Initialize the stack of available indexes.
I = sub2ind1(x0);

% Initialize the distance to +?, excepted for the boundary conditions.
D = zeros(n)+Inf; 
D(I) = 0;

% Initialize the state to 0 (unexplored), excepted for the boundary point S (indexed by |I|) to 1 (front).
S = zeros(n);
S(I) = 1;

% The first step of each iteration of the method is to pop the from stack the element i
% with smallest current distance Di.

[tmp,j] = sort(D(I)); j = j(1);
i = I(j); I(j) = [];

% We update its state S
% to be dead (-1).

S(i) = -1;

% Retrieve the list of the four neighbors.
J = [Neigh(i,1); Neigh(i,2); Neigh(i,3); Neigh(i,4)];

% Remove those that are dead (no need to consider them anymore).
J(S(J)==-1) = [];

% Add those that are not yet considered (state 0) to the front stack I (state 1).
J1 = J(S(J)==0);
I = [I; J1];
S(J1) = 1;

% Update neighbor values. For each neightbo j
% of i, perform the update, assuming the length of the edge between j and k is Wj.
% Dj?mink?jDk+Wj.

for j=J'
    dx = min( D([Neigh(j,1) Neigh(j,2)]) );
    dy = min( D([Neigh(j,3) Neigh(j,4)]) );
    D(j) = min(dx+W(j), dy+W(j));
end

%% Exercice 1

% Implement the Dijkstra algorithm by iterating these step while the stack |I| is non empty. Display from time to time the front that propagates.







% Display the geodesic distance map using a cosine modulation to make the level set appears more clearly.
displ = @(D)cos(2*pi*5*D/max(D(:)));
clf;
imageplot(displ(D));
colormap jet(256);


%% Fast Marching
% The Dijstra algorithm suffers from a strong metrization problem, and it actually computes the ?1
% distance on the grid.
% The Fast Marching algorithm replace the graph update by a local resolution of the Eikonal equation. This reduces significantly the grid bias, and can be shown to converge to the underlying geodesic distance when the grid step size tends to zero.
% Over a continuous domain, the distance map D(x)
% to a set of seed points S is the unique solution in the viscosity sense
% ?x?S,??D(x)?=W(x)and?y?S,D(y)=0.
% The equation is then discretized on a grid of n×n
% pixel, and a solution (Dk,?)nk,?=1?Rn×n is found by using an upwind finite difference approximation, that is faithful to the viscosity solution
% ?(k,?)?S~,?(?D)k,??=Wk,?$and?(k,?)?S~,Dk,?=0,
% where S~ is the set of discrete starting points (defined here by |x0|).
% To be consisten with the viscosity solution, one needs to use a non-linear upwind gradient derivative. This corresponds to computing the norm of the gradient as
% ?(?D)k,??2=max(Dk+1,??Dk,?,Dk?1,??Dk,?,0)2+max(Dk,?+1?Dk,?,Dk,??1?Dk,?,0)2.
% A each step of the FM propagation, one update Dk,??d
% by solving the eikonal equation with respect to Dk,? alone. This is equivalent to solving the quadratic equation
% (d?dx)2+(d?dy)2=w2wherew=Wk,?.
% and where dx=min(Dk+1,?,Dk?1,?)anddy=min(Dk,?+1,Dk,??1).
% The update is thus defined as
% d={dx+dy+??2when??0,min(dx,dy)+wotherwise.$$where?=2w2?(dx?dy)2.
% Note that in the case where ?<0
% , one has to use the Dijkstra update.
% Once the Dijstra algorithm is implemented, the implementation of the Fast Marching is trivial. It just corresponds to replacing the graph udpate

D(j) = min(dx+W(j), dy+W(j));

% by the eikonal update.

Delta = 2*W(j) - (dx-dy)^2;
if Delta>=0
    D(j) = (dx+dy+sqrt(Delta))/2;
else
    D(j) = min(dx+W(j), dy+W(j));
end

%% Exercice 2
% Implement the Fast Marching algorithm. Display from time to time the front that propagates.












% Display the geodesic distance map using a cosine modulation to make the level set appears more clearly.

clf;
imageplot(displ(D));
colormap jet(256);

%% Computation of Geodesic Paths

% We use a more complicated, non-constant metric, with a bump in the middle.

n = 100;
x = linspace(-1,1,n);
[Y,X] = meshgrid(x,x);
sigma = .2;
W = 1 + 8 * exp(-(X.^2+Y.^2)/(2*sigma^2));

clf;
imageplot(W);


x0 = round([.1;.1]*n); %  [.8;.8]]*n);

%% Exercice 3
% Compute the distance map to these starting point using the FM algorithm.

% non periodic boundary conditions
boundary = @(x)x.*(x<=n & x>0) + (2-x).*(x<=0) + (2*n-x).*(x>n);






% Once the geodesic distance map to S has been computed, the geodesic curve between any point x1 and S extracted through gradient descent
% ??(t)=??t?D(?(t))and?(0)=x1 where ?t>0
% controls the parameterization speed of the resulting curve.
% To obtain unit speed parameterization, one can use ?t=??D(?(t))??1
% (one need to be careful when ? approaches S since D is not smooth on S).
% Compute the gradient G0(x)=?D(x)?R2
% of the distance map. Use centered differences.

options.order = 2;
G0 = grad(D, options);

% Normalize the gradient to obtained G(x)=G0(x)/?G0(x), in order to have unit speed geodesic curve (parameterized by arc length).

G = G0 ./ repmat( sqrt( sum(G0.^2, 3) ), [1 1 2]);



% The geodesic is then numerically computed using a discretized gradient descent, which defines a discret curve (?k)k using
% ?k+1=?k??G(?k)
% where ?k?R2 is an approximation of ?(t) at time t=k?, and the step size ?>0 should be small enough.
% Step size ? for the gradient descent.

tau = .8;

% Initialize the path with the ending point.

x1 = round([.9;.88]*n);
gamma = x1;

% Define a shortcut to interpolate G
% at a 2-D points. Warning: the |interp2| switches the role of the axis ...

Geval = @(G,x)[interp2(1:n,1:n,G(:,:,1),x(2),x(1)); ...
             interp2(1:n,1:n,G(:,:,2),x(2),x(1)) ];

% Compute the gradient at the last point in the path, using interpolation.

g = Geval(G, gamma(:,end));

% Perform the descent and add the new point to the path.

gamma(:,end+1) = gamma(:,end) - tau*g;

%% Exercice 4

% Perform the full geodesic path extraction by iterating the gradient descent. 
% You must be very careful when the path become close to x0, because the distance function is not differentiable at this point. You must stop the iteration when the path is close to x0.










% Display the geodesic curve.


clf; hold on;
imageplot(W); colormap gray(256);
h = plot(gamma(2,:),gamma(1,:), '.b'); set(h, 'LineWidth', 2);
h = plot(x0(2),x0(1), '.r'); set(h, 'MarkerSize', 25);
h = plot(x1(2),x1(1), '.b'); set(h, 'MarkerSize', 25);
axis ij;




