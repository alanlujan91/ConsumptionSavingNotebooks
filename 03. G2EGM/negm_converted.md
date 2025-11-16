Computational Economics (2021) 58:747–775
https://doi.org/10.1007/s10614-020-10045-x

A Guide on Solving Non-convex Consumption-Saving
Models

Jeppe Druedahl1

Accepted: 4 September 2020 / Published online: 26 October 2020
© Springer Science+Business Media, LLC, part of Springer Nature 2020

Abstract
Consumption-saving models with adjustment costs or discrete choices are typically
hard to solve numerically due to the presence of non-convexities. This paper provides
a number of tools to speed up the solution of such models. Firstly, I use that many
consumption models have a nesting structure implying that the continuation value
can be efﬁciently pre-computed and the consumption choice solved separately before
the remaining choices. Secondly, I use that an endogenous grid method extended
with an upper envelope step can be used to solve efﬁciently for the consumption
choice. Thirdly, I use that the required pre-computations can be optimized by a novel
loop reordering when interpolating the next-period value function. As an illustrative
example, I solve a model with non-durable consumption and durable consumption
subject to adjustment costs. Combining the provided tools, the model is solved almost
50 times faster than with standard value function iteration for a given level of accuracy.
Software is provided in both Python and C++.

Keywords Endogenous grid method · Post-decision states · Stochastic dynamic
programming · Continuous and discrete choices · Occasionally binding constraints

JEL Classiﬁcation C6 · D91 · E21

1 Introduction

Multi-dimensional consumption-saving models with adjustment costs or discrete
choices are typically hard to solve numerically due to the presence of non-convexities.
Starting from value function iteration (VFI), which is simple and straightforward, but

B Jeppe Druedahl

jeppe.druedahl@econ.ku.dk

1

CEBI, Department of Economics, University of Copenhagen, Øster Farimagsgade 5, Building 26,
1353 Copenhagen K, Denmark

123

748

J. Druedahl

inherently slow, this paper provides a guide on reducing computational time in such
models using three layers of optimization.

In the ﬁrst layer, I use that many consumption saving models have a nesting structure
such that the continuation value can be efﬁciently pre-computed and the consumption
choice solved separately before the remaining choices. I refer to this as the nested
value function iteration (NVFI).

In the second layer, I use that the nested consumption problem under weak assump-
tions can be solved efﬁciently by using an extension of the endogenous grid method
(EGM) originally developed by Carroll (2006) for one dimensional models.1 I refer
to this as the nested endogenous grid method (NEGM). This step relies on a one-
dimensional version of the multi-dimensional upper envelope algorithm developed in
Druedahl and Jørgensen (2017).

In the third layer of optimization, I use that the pre-computation of the continuation
value needed in both NVFI and NEGM can be computed efﬁciently by introducing a
novel loop reordering reducing the number of computations and improving memory
access. I refer to the improved solution methods as NVFI+ and NEGM+.

To study the implications for speed and accuracy of the proposed optimizations, I use
a buffer-stock consumption-saving model with non-durable and durable consumption
and adjustment costs similar to Berger and Vavra (2015). I show that NEGM+ relative
to VFI provides a speed-up factor of almost 50 for a given level of accuracy. The
speed-up is reduced to a factor of 15 without the optimized interpolation approach.
NVFI provides a speed-up factor of about 10.5, increasing to 13.5 with the optimized
interpolation approach.

These results are furthermore obtained when using a non-robust version of VFI,
where the underlying non-convex optimization problems are solved with a local solver
without any multi-start, which therefore could end up in local maxima. I discuss
how this speed-up can be expected to vary with both the model speciﬁcation and
implementation choices. Similar results are obtained for an extended benchmark model
with two durable stocks adding both a state and a choice to the model.

Code libraries to implement the proposed solution algorithms are provided in both

a C++ version, and a somewhat slower, but more accessible Python version.2
Related literature Existing extensions of the endogenous grid method does either
not consider non-convexities (Barillas and Fernández-Villaverde 2007; Hintermaier
and Koeniger 2010; White 2015; Ludwig and Schön 2018) or restrict attention to
the one-dimensional case (Fella 2014; Iskhakov et al. 2017). This paper adds to the
previous extensions of the endogenous grid method by simultaneously handling multi-
dimensional models and non-convexities.

The only other paper, which provides a EGM algorithm to solve multi-dimensional
models with non-convexities is Druedahl and Jørgensen (2017). Their G2EGM algo-
rithm, however, requires that the equation system of stacked discrepancies in the ﬁrst
order conditions and post-decision state equations have a unique solution. This can
be cumbersome to determine for a given model, and small model changes can alter

1 Jørgensen (2013) and Low and Meghir (2017) shows that EGM is orders of magnitude faster than VFI
in one dimensional models.
2 The code is available at github.com/NumEconCopenhagen/ConsumptionSavingNotebooks. The Python
code is optimized with just-in-time compilation from the Numba package.
123

A Guide on Solving Non-convex Consumption-Saving Models

749

whether G2EGM is applicable or not (see their discussion in Sect. 5.4). By compari-
son, the NEGM only uses the ﬁrst order condition for consumption, and it is thus more
generally applicable. As an example, the benchmark model used in this paper cannot
be solved by G2EGM. The modularity of NEGM is an additional beneﬁt relative to
G2EGM. The algorithms for computing the upper envelope and speeding up the inter-
polation step presented in this paper can be re-used across different models with little
(or no) modiﬁcation. The code thus becomes more generic and easier to understand
and debug. For models where both G2EGM and NEGM can be applied, the former will
most likely be faster, if the fast interpolation scheme proposed in this paper is used,
because it completely avoids any VFI step. However, G2EGM typically requires com-
puting additional value function derivatives, and the multi-dimensional upper envelope
is more costly than the one-dimensional upper envelope used in NEGM. I show that in
the case of the model from Druedahl and Jørgensen (2017) the G2EGM is only about
25% faster than NEGM in their baseline speciﬁcation, but somewhat more accurate.
In speciﬁcations with uncertainty where numerical integration is required the result
can ﬂip.

The idea of pre-computing the continuation value, which EGM fundamentally relies
on, is well-known in the operations research and engineering literature. For inﬁnite
horizon models it is thus common to iterate on a Bellman equation in the post-decision
value function; see in particular Van Roy et al. (1997), Powell (2011) and Bertsekas
(2012).3 The idea of using pre-computations to limit the costs of numerical integration
was also proposed by Judd et al. (2017) in an algorithm where the value function is
approximated by a parametric function. The idea of reformulating the model using its
nesting structure is similar to the general idea of plan factorization discussed in Ma
and Stachurski (2018).

Structure
The paper is henceforth structured as follows. Section 2 presents the
solution methods in light of a speciﬁc model. Section 3 presents speed and accuracy
results comparing the proposed solution methods with VFI. Section 4 presents the
general model class where the proposed solution methods can be used. Section 5
provides a comparison with G2EGM. Section 6 concludes.

2 Solution Method

To ﬁx ideas, consider a buffer-stock consumption-saving model with non-durable
and durable consumption similar to Berger and Vavra (2015) and Harmenberg and
Oberg (2017). The household’s state variables are cash-on-hand mt , the stock of the
durable good nt , and the persistent component of income pt . Each period the household
chooses consumption ct and durable consumption dt . Per period utility is CRRA over
a Cobb–Douglas aggregate,

u(ct , dt ) =

(cα
t

(dt + d)1−α)1−ρ
1 − ρ

, α ∈ (0, 1), ρ > 0,

(2.1)

3 See Hull (2015) for some extensions and an application in economics.

123

750

J. Druedahl

where d ≥ 0 is a ﬂoor under durable consumption. Future utility is discounted with
a factor β > 0. The household’s income, yt , follows an exogenous stochastic process
with persistent and fully transitory shocks,

pt+1 = ψt+1 pλ
t

,

log ψt+1 ∼ N (−0.5σ 2

ψ , σ 2

ψ ), λ ∈ (0, 1]

yt+1 = ξt+1 pt+1,

log ξt+1 ∼ N (−0.5σ 2

ξ , σ 2

ξ ).

(2.2)

(2.3)

(cid:5)= nt ) it incurs a
If the household adjusts its stock of the durable good (i.e. dt
proportional adjustment cost τ ∈ (0, 1). It’s cash-on-hand after selling the beginning-
of-period stock of the durable good, xt , is therefore

xt = mt + (1 − τ )nt .

The household’s end-of-period assets, at , is therefore

(cid:2)

at =

mt − ct
xt − ct

if dt = nt
if dt (cid:5)= nt

.

(2.4)

(2.5)

Further, the household cannot borrow, at ≥ 0, and the interest rate on savings is r .
Deﬁne R = 1 + r . The next-period cash-on-hand therefore is

mt+1 = Rat + yt+1.

Finally, the durable stock depreciates with the rate of δ ∈ (0, 1), i.e.

nt+1 = (1 − δ)dt .

(2.6)

(2.7)

2.1 Bellman Equation

The household’s value function can be written as a maximum over the value function
when not adjusting, vkeep

, and the value function when adjusting vad j.

, i.e.

t

t

vt ( pt , nt , mt ) = max{vkeep
s.t.

t

( pt , nt , mt ), vad j.

t

( pt , xt )}

xt = mt + (1 − τ )nt ,

(2.8)

where xt is cash-on-hand after selling the beginning-of-period stock of the durable
good. The value function when keeping is

vkeep
t

( pt , nt , mt ) = max

ct

u(ct , nt ) + βEt [vt+1( pt+1, nt+1, mt+1)]

s.t.

123

A Guide on Solving Non-convex Consumption-Saving Models

751

at = mt − ct
mt+1 = Rat + yt+1
nt+1 = (1 − δ)nt

at ≥ 0.

The value function when adjusting is

vad j.
t

( pt , xt ) = max
ct ,dt

s.t.

u(ct , dt ) + βEt [vt+1( pt+1, nt+1, mt+1)]

at = xt − ct − dt
mt+1 = Rat + yt+1
nt+1 = (1 − δ)dt

at ≥ 0.

(2.9)

(2.10)

This problem can be solved straightforwardly with a value function iteration (VFI)
algorithm.4 The keeper problem in Eq. (2.9) is costly because the state space is multi-
dimensional, and the adjuster problem in Eq. (2.10) is costly because the decision space
is multi-dimensional. The main computational cost in both problems, however, is the
computation of the continuation value for each new guess of the optimal choice(s).
The continuation value, Et [vt+1( pt+1, nt+1, mt+1)], must be computed using some
form of numerical integration, which will always require multiple interpolations of
the next-period value function.

2.2 Nesting

In order to speed up the value function iteration algorithm it is beneﬁcial to use some
nesting structure present in the model.

Firstly, note that knowing the so-called post-decision states, i.e. the persistent com-
ponent of income pt , the stock of the durable good dt and end-of-period assets at , is
sufﬁcient for computing the continuation value. In particular, knowing consumption
ct is irrelevant for computing the continuation value when already conditioning on the
post-decision states. This leads me to specify the post-decision value function

wt ( pt , dt , at ) = βEt [vt+1( pt+1, nt+1, mt+1)],

(2.11)

and re-write the keeper problem as

vkeep
t

( pt , nt , mt ) = max

ct

u(ct , nt ) + wt ( pt , dt , at )

s.t.

at = mt − ct ≥ 0
dt = nt .

4 An alternative would be time iteration, see Rendahl (2015).

(2.12)

123

752

J. Druedahl

Secondly, note that we are always allowed to view the adjuster problem as sequential.
We can therefore imagine that the household ﬁrst chooses how much to buy of the
durable good and then afterwards chooses consumption. The consumption choice is
then exactly the same as for a keeper starting the period with the chosen stock of
the durable good for some level of cash-on-hand. We can thus re-write the adjuster
problem as5

vad j.
t

( pt , xt ) = max
dt

vkeep
t

( pt , dt , mt )

s.t.

mt = xt − dt
dt ∈ [0, xt ].

(2.13)

Taken together this double nesting structure allows us to specify the following nested
value function iteration (NVFI) with the following three steps in each time period:

Nested Value Function Iteration (NVFI)

Step 1

Step 2

Compute the post-decision value function wt in Eq. (2.11) on a grid over the post-decision

states pt , dt , and at

Solve the keeper problem in Eq. (2.12) on a grid over the pre-decision states pt , nt , and mt

using interpolation of the post-decision value function wt computed in step 1

Step 3

Solve the adjuster problem in Eq. (2.13) using interpolation of the keeper value function

found in step 2

The nested value function iteration algorithm provides a speed up relative to the stan-
dard value function iteration algorithm for two reasons. Firstly, it replaces multiple
interpolations of the next-period value function during numerical integration of the
continuation value with a single interpolation of the post-decision value function when
solving the keeper problem, plus a pre-computation step to construct the post-decision
value function itself. Even allowing the post-decision grid to be relatively dense, to
limit the loss in precision from the additional interpolation layer, this implies a lot
fewer interpolations.6

Secondly, it reduces the dimensionality of the decision problem for the adjuster,
and replaces evaluating the utility function and multiple evaluations of the next-period
value function, with a single interpolation of the value function for the keeper.

5 An alternative is to instead interpolate the keepers consumption function and then calculate the implied
value of choice by evaluating the utility function and the post-decision value function. This could in principle
improve precision, but the effect seems minor in practice.
6 Assume that K evaluations of the value-of-choice is needed when solving the keeper problem for each
node in the state space, and Q is the number of integration nodes needed to calculate the expectation in the
continuation value. In NVFI we then need K + ϑ Q interpolations for each node in the state grid, where ϑ
is the ratio of nodes in the post-decision grid relative to the state grid. In VFI we need K Q interpolations
for each node. We have K Q > K + ϑ Q ⇔ ϑ < K (1 − Q−1), which for reasonable Q roughly implies
that fewer interpolations is needed whenever ϑ < K .

123

A Guide on Solving Non-convex Consumption-Saving Models

753

2.3 EGM with an Upper Envelope

The bottleneck in the NVFI proposed above can be shown (see Sect. 3) to be the
solution of the keeper problem due to the multi-dimensional state space. Since it is
a pure consumption problem, it can however be solved by using an endogenous grid
method (EGM) instead of a numerical solver for each node in the state grid.

The general

idea in EGM, originally developed in Carroll (2006) for one-
dimensional models without non-convexities, is to ﬁx the end-of-period asset state
and then use the Euler-equation and the budget constraint to infer respectively the
consumption choice and the level of cash-on-hand. For non-convex multi-dimensional
models several complications arise. Firstly, the non-convexity of the decision problem
due to the adjustment cost implies that the value function is not globally concave and
that the ﬁrst order conditions are not sufﬁcient. By a standard variational argument
the Euler-equation is, however, still necessary. Secondly, the multi-dimensionality of
the model implies that interpolation to a regular grid is required because interpolation
of irregular grids are very costly in multiple dimensions.7 A parsimonious solution
to both problems is to use a one dimensional version of the multi-dimensional upper
envelope algorithm developed in Druedahl and Jørgensen (2017).8

In the present model, the Euler-equation is given by

α(1−ρ)−1
uc(ct , dt ) = αc
t

(dt + d)(1−α)(1−ρ) = qt ( pt , dt , at ),

(2.14)

where qt ( pt , dt , at ) is the post-decision marginal value of cash,9

qt ( pt , dt , at ) ≡ β REt [uc(ct+1, dt+1)]
= β REt [αc

α(1−ρ)−1
t+1

(dt+1 + d)(1−α)(1−ρ)].

(2.15)

By solving the Euler-equation for ct , this lets us deﬁne the following functions from
post-decision states to consumption and then cash-on-hand,

(cid:3)

qt ( pt , dt , at )
α(dt + d)(1−α)(1−ρ)

(cid:4) 1

α(1−ρ)−1

ct = z( pt , at , dt ) =

mt = at + ct .

(2.16)

(2.17)

Whenever the Euler-equation is necessary all points on the consumption function can
be generated from Eqs. (2.16)–(2.17) starting from some values of the post-decision
states. When the Euler-equation is not also sufﬁcient, some of the points created
will, however, not be a point on the consumption function. Such non-optimal points

7 See the detailed discussion in Ludwig and Schön (2018) regarding triangulazation and Delaunay inter-
polation.
8 The ﬁrst upper envelope algorithms were developed by Fella (2014) and Iskhakov et al. (2017) for one
dimensional models, and were thus not designed to deliver regular grids in multi-dimensional models.
9 Due to the envelope theorem, we have that vt,m ( pt , nt , mt ) = uc(ct , dt ), and therefore we could instead
have used the deﬁnition qt ( pt , dt , at ) ≡ wt,a ( pt , dt , qt ) = β REt [vt+1,m ( pt+1, nt+1, mt+1)].

123

754

J. Druedahl

must therefore be disregarded. Algorithm 1 solves this task and returns the optimal
consumption choices on an exogenously chosen grid of cash-on-hand. The inputs are
exogenously chosen values of the persistent component of income p and the stock of
the durable good d, and exogenous vectors of end-of-period assets a and beginning-
of-period cash-on-hand m.

Lines 1–2 initialize the optimal value v at negative inﬁnity. Lines 3–6 apply Eqs.
(2.11) and (2.16)–(2.17) for all points in the input end-of-period asset vector. Lines
7–10 use that the borrowing constraint is binding for all cash-on-hand levels lower
than the cash-on-hand level implied by assuming no saving (i.e. a1 = 0). Line 11 starts
a loop over neighboring points in the endogenously created cash-on-hand grid, while
the loop in line 12 is over each point in the exogenously chosen cash-on-hand grid. If a
point in the exogenous grid is between two neighboring points in the endogenous grid
(line 13) the implied linear interpolated value of consumption (line 14) and value-of-
choice (line 15) is calculated. If the value-of-choice is the best yet found (line 16) the
optimal consumption and value entries are updated (lines 17–18). For dense enough
grids, the endogenously created points with non-optimal value of consumption are
thus disregarded because they imply lower values-of-choice.10

With the upper envelope algorithm in hand the nested endogenous grid method

(NEGM) is given by the following three steps:

Nested endogenous grid method (NEGM)

Step 1

Compute the post-decision functions wt and qt in Eqs. (2.11) and (2.16) on a grid over the

post-decision states pt , dt and at

Step 2

Solve the keeper problem in Eq. (2.12) on a grid over the pre-decision states pt , nt , and mt ,

where the combined EGM and upper envelope in Algorithm 1 is applied for each
combination of pt and nt

Step 3

Solve the adjuster problem in Eq. (2.13) using interpolation of the keeper value function

found in step 2

NEGM preserves all the beneﬁts of NVFI and speeds up the solution of the keeper
problem by using EGM. Note that the upper envelope part of Algorithm 1 (lines 7–
18) is a cheap operation as it involves only simple logical and algebraic operations.
Furthermore, Algorithm 1 is completely general in the sense that it does not depend
on any speciﬁc structure of the model considered here, and can easily be extended to
incorporate e.g. additional post-decision states.

2.4 Fast Interpolation

The main bottleneck in NEGM can be shown (see Sect. 3) to be the computation of
the post-decision functions wt ( pt , dt , at ) and qt ( pt , dt , at ) in step 1. Again, however,
there is some structure in the problem, which can be used for speed up.

To see this, ﬁrst note that multi-linear interpolation for a single point can be com-
puted as described in Algorithm 2. This algorithm works by ﬁrst ﬁnding the grid

10 See Druedahl and Jørgensen (2017) for additional details.

123

A Guide on Solving Non-convex Consumption-Saving Models

755

Algorithm 1: EGM and Upper Envelope

input:

persistent component of income: p
stock of durable good: d
cash-on-hand grid vector: Gm = {m j }#m
j=1
end-of-period asset grid vector: Ga = {ai }#a
i=1

, m1 = 0

, a1 = 0

5

1 for j ∈ {1, 2, . . . , #m } do
v j = −∞
2
3 for i ∈ {1, 2, . . . , #a } do
4

wi = w( p, d, ai )
ci = z( p, d, ai )
mi = ai + ci
6
7 for j ∈ {1, 2, . . . , #m } do
8
9

if m j ≤ m1 then
c j = m j
v j = u(c j , d) + w1

10

11 for i ∈ {1, 2, . . . , #a − 1} do
for j ∈ {1, 2, . . . , #m } do
12
13

14

15

16

17

18

ci
j
vi
j
if vi
j

if m j ∈ [mi , mi+1] then
= ci + ci+1−ci
(m j − mi )
mi+1−mi
(cid:5)
wi + wi+1−wi
= u(ci
, d) +
ai+1−ai
j
> v j then
v j = vi
j
c j = ci
j

((m j − ci
j

(cid:6)
) − ai )

19 return v1, v2, . . . , v#m

, c1, c2, . . . , c#m

positions in each dimension for the point to be interpolated (lines 1–3); e.g. by binary
search. Secondly, the interpolated value is calculated as the weighted sum of the func-
tion values at the corners of the hypercube surrounding the point to be interpolated
(lines 4–12).

The standard approach to calculate the wt and qt functions is to ﬁrst ﬁx the post-
decision states ( pt , nt , at ) and then use multi-linear interpolation of the next-period
value function vt+1( pt+1, nt+1, mt+1) for all nodes in a grid of the shocks (ψt+1,
ξt+1) and then take the appropriately weighted sum of the interpolated values. This is
formalized in Algorithm 3.

We know, however, that we for each combination of pt , nt , ψt+1 and ξt+1 need to
interpolate the next-period value function for a vector of at values. This implies that
we for given pt+1 and nt+1 need to interpolate the next-period value function for a
vector of mt+1 values. When the vector for at is chosen as monotonically increasing,
the implied vector for mt+1 will also be monotonically increasing for given ψt+1
and ξt+1. This implies that the interpolation can be done with Algorithm 4. This is
beneﬁcial for three reasons: Firstly the search in the p and n dimensions (lines 1–2)

123

756

J. Druedahl

is done once instead of for each m. Secondly, the search for each a is faster because
of the ordering (lines 3–9). Thirdly, the memory access when looking up the value of
the next-period value function (line 19) is often made at neighboring indices due to
the ordering, which implies fewer cache misses.

Algorithm 5 shows (see lines 12–13) how to use Algorithm 4 when computing
the w and q functions using a reordering of the loops such that the loop over a is
inside the loops over the shocks ψ and ξ . Algorithm 5 returns exactly the same results
as Algorithm 3, and is only slightly more complicated to code. Note also that the
vectorized interpolation approach in Algorithm 4 does not depend on any speciﬁc
structure of the model considered here, and can be straightforwardly extended to
higher dimensional problems.11

Algorithm 2: Linear Interpolation (interp)

input:

grid vectors: G p = { p}# p
array of known values at tensor product of grid vectors: v[:, :, :]
point to interpolate for: ( p, n, m)

, Gm = {m}#m

, Gn = {n}#n

jm =1

j p=1

jn =1

, p j p +1)
, n jn +1)

, m jm +1)

1 Find j p such that p ∈ [ p j p
2 Find jn such that n ∈ [n jn
3 Find jm such that m ∈ [m jm
4 Initialize ˆv = 0
5 Calculate Ω = ( p j p +1 − p j p
6 for k p ∈ {0, 1} do
7

)(n jn +1 − n jn

)(m jm +1 − m jm

)

if k p = 0 then ω p = p j p +1 − p else ω p = p − p j p
for kn ∈ {0, 1} do

if kn = 0 then ωn = n jn +1 − n else ωn = n − n jn
for km ∈ {0, 1} do

if km = 0 then ωm = m jm +1 − m else ωm = m − m jm
ˆv +=

V [ j p + k p, jn + kn , jm + km ]

ωm ωn ω p
Ω

8
9

10
11

12

13 return ˆv

2.5 Implementation Details

This sub-section explains the implementations choices made in the underlying code.

Parametrization The chosen parameters are β = 0.965, ρ = 2, α = 0.9 d = 10−2,
R = 1.03, τ = 0.10, δ = 0.15, σψ = σξ = 0.1, and λ = 1.

Grids The grid for p is constructed by the recursion:

p1 = p

11 A slight further speed-up is possible by using that the interpolations in lines 12–13 in Algorithm 5 is for
the same vector of next-period states. Lines 1–9 of Algorithm 4 can therefore be skipped when calling the
interpolation algorithm for the second time in line 13 of Algorithm 5.

123

A Guide on Solving Non-convex Consumption-Saving Models

757

Algorithm 3: Post-decision functions: Standard approach

input:

, Gm = {m}#m

, Gn = {n}#n

grid vectors: G p = { p}# p
j p=1
shock vectors: Gψ = {ψ}#ψ
shock weight vectors: Gψ w = {ψ w}#ψ
next-period value function: v+[:, :, :]
next-period marginal utility of consumption: uc,+[:, :, :]

jξ =1
, Gξ = {ξ w}#ξ

jn =1
, Gξ = {ξ }#ξ

jψ =1

jψ =1

jξ =1

jm =1

, Ga = {a}#a

ja =1

for jn ∈ {1, 2, . . . , #n } do

1 Initialize w[:, :, :] = 0
2 Initialize q[:, :, :] = 0
3 for j p ∈ {1, 2, . . . , # p} do
4
5
6
7
8

for ja ∈ {1, 2, . . . , #a } do

for jψ ∈ {1, 2, . . . , #ψ } do

for jξ ∈ {1, 2, . . . , #ξ } do

ψ jψ

p+ = p j p
n+ = (1 − δ)n jn
y+ = p+ξ jξ
m+ = Ra ja
ˆv+ = inter p(G p, Gn , Gm , v+, ( p+, n+, m+))
ˆuc,+ = inter p(G p, Gn , Gm , uc,+, ( p+, n+, m+))
ξ w
w[ j p, jn , ja ] += βψ w
jξ
jψ
ξ w
q[ j p, jn , ja ] += β Rψ w
jξ
jψ

ˆv+
ˆuc,+

+ y+

9

10

11

12
13
14

15

16 return w, q

pi = pi−1 +

p − pi−1
(# p − (i − 1))1.1

, i = 2, 3, . . . , # p,

with p = 10−4, p = 3 and # p = 150. The other grids are constructed similarly with
n = 0, n = 3, #n = 150, m = 0, m = 10, #m = 300, x = 0, x = 13, #x = 300,
a = 0, a = 11, and #a = 300.12 Extrapolation outside of the grids for p and n are not
allowed.

Interpolation Deﬁne the following negative inverse transformations of the value
functions and marginal utilities of consumption:

˜vkeep
t

( pt , nt , mt ) ≡ −

1

vkeep
t

( pt , nt , mt )
1
( pt , xt )

˜vad j.
t

( pt , xt ) ≡ −

vad j.
t

12 The value for x = m + n ensures that the grid is wide enough for comparing the adjust and keep cases.
The value for a < x is chosen somewhat lower because the household will always choose to consume some
it’s resources.

123

758

J. Druedahl

Algorithm 4: Linear interpolation for monotone vector (interpvec)

input:

}# p
j p =1

}#n
jn =1

, Gn = {n jn

, Gm = {m jm

grid vectors: G p = { p j p
array of known values at tensor product of grid vectors: v[:, :, :]
ﬁrst two dimensions in points to interpolate for: p, n
last dimension in points to interpolate for:
m1, m2, . . . , m# where m1 < m2 < · · · < m#
, p j p +1)
, n jn +1)

}#m
jm =1

1 Find j p such that p ∈ [ p j p
2 Find jn such that n ∈ [n jn
3 for i ∈ {1, 2, . . . , #} do
if i = 1 then
4
Find j 1
5

6

7
8

9

else

= j i−1
j i
m
m
while mi ≥ m
j i
+= 1
m

do

j i
m +1

m such that m1 ∈ [m j 1

, m j 1

m +1

)

m

10 Initialize ˆvi = 0 for i ∈ {1, 2, . . . , #}
11 for k p ∈ {0, 1} do
12

if k p = 0 then ω p = p j p +1 − p else ω p = p − p j p
for kn ∈ {0, 1} do

if kn = 0 then ωn = n jn +1 − n else ωn = n − n jn
for i ∈ {1, 2, . . . , #} do

Calculate Ω = ( p j p +1 − p j p
for km ∈ {0, 1} do

)(n jn +1 − n jn

)(m

j i
m +1

− m

)

j i
m

if km = 0 then ωm = mi
ωm ωn ω p
ˆvi +=
Ω

jm +1
V [ j p + k p, jn + kn , j i
m

− m else ωm = m − m

+ km ]

j i
m

13
14

15
16

17
18

19

20 return ˆv1, ˆv2 . . . , ˆv#

˜ukeep
c,t

( pt , nt , mt ) ≡

1

uc(ckeep
t

˜uad j.
c,t

( pt , xt ) ≡

uc(cad j.

t

( pt , nt , mt ), nt )
1
( pt , xt ), dad j.

t

.

( pt , xt ))

These transformed functions are always positive and increasing, and satisfy

˜vkeep
t

( pt , nt , mt ) = 0

lim
mt →0

lim
xt →0
˜ukeep
c,t

lim
mt →0

˜vad j.
t

( pt , xt ) = 0

( pt , nt , mt ) = 0

˜uad j.
c,t

( pt , xt ) = 0,

lim
xt →0

123

A Guide on Solving Non-convex Consumption-Saving Models

759

Algorithm 5: Post-decision functions: Reordered loops

input:

, Gm = {m}#m

, Gn = {n}#n

grid vectors: G p = { p}# p
j p=1
shock vectors: Gψ = {ψ}#ψ
shock weight vectors: Gψ w = {ψ w}#ψ
next-period value function: v+[:, :, :]
next-period marginal utility of consumption: uc,+[:, :, :]

jξ =1
, Gξ = {ξ w}#ξ

jn =1
, Gξ = {ξ }#ξ

jψ =1

jψ =1

jξ =1

jm =1

, Ga = {a}#a

ja =1

1 Initialize w[:, :, :] = 0
2 Initialize q[:, :, :] = 0
3 for j p ∈ {1, 2, . . . , # p} do
4
5
6
7

for jn ∈ {1, 2, . . . , #n } do

for jψ ∈ {1, 2, . . . , #ψ } do

for jξ ∈ {1, 2, . . . , #ξ } do

ψ jψ

p+ = p j p
n+ = (1 − δ)n jn
y+ = p+ξ jξ
for ja ∈ {1, 2, . . . , #a } do
+ = Ra ja

+ y+

m ja

8

9

10

11

12

13

14

15

16

17 return w, q

ˆv+ = inter pvec(G p, Gn , Gm , v+, ( p+, n+, m1
+, m2
ˆuc,+ = inter pvec(G p, Gn , Gm , uc,+, ( p+, n+, m1
for ja ∈ {1, 2, . . . , #a } do
w[ j p, jn , ja ] += ψ w
jψ
q[ j p, jn , ja ] += ψ w
jψ

ˆv ja
+
ˆu ja
c,+

ξ w
jξ
ξ w
jξ

+, . . . , m#a
+ )
+, . . . , m#a
+, m2
+ )

which is beneﬁcial when interpolating because no inﬁnities are involved.

To improve on precision the post-decision value function are calculated as

wt ( pt , dt , at ) = βEt [vt+1( pt+1, nt+1, mt+1)]
⎧
⎡
⎨

= βEt

⎣

⎩

− 1
˜vkeep
t+1
− 1
˜vad j
t+1

if ˜vkeep
t+1
if ˜vkeep
t+1

≥ ˜vad j
t+1
< ˜vad j
t+1

⎤

⎦ ,

t+1 and ˜vad j.

where ˜vkeep
value of cash function is calculated as

t+1 are interpolated separately. Similarly, the post-decision marginal

qt ( pt , dt , at ) = β REt [uc(ct ( pt , nt , mt ), dt ( pt , nt , mt ))]
− 1
˜ukeep
c,t+1
− 1
˜uad j
c,t+1

≥ ˜vad j
t+1
< ˜vad j
t+1

if ˜vkeep
t+1
if ˜vkeep
t+1

= β REt

⎧
⎨

⎦ ,

⎩

⎣

⎤

⎡

123

760

J. Druedahl

where ˜ukeep

c,t

and ˜uad j.

c,t are interpolated separately.

The required changes to lines 12–15 of Algorithm 3 and lines 12–16 of Algorithm

5 are straightforward.

All numerical integration is done with Gauss–Hermite quadrature using 5 nodes

for both the persistent and the transitory shocks.

Simulation The initial values in the simulation are drawn as:

log p0 ∼ N (log(1), 0.2)
log d0 ∼ N (log(0.8), 0.2)
log a0 ∼ N (log(0.2), 0.1).

3 Speed and Accuracy

To assess the speed and accuracy of the proposed solution methods, I compare them
with a standard value function iteration. I solve the model backwards for T = 50
periods (iterations) using the implementation settings described in Sect. 2.5.

Following Judd (1992) and Santos (2000), I measure accuracy as the log10 of the
relative absolute Euler error across households optimally making an interior consump-
tion choice. Speciﬁcally, I use a simulation sample of N = 100, 000 and calculate
(cid:14)

(cid:14)

T
t=1
(cid:14)
T
t=1

N
Eit 1ait >(cid:14)
i=1
(cid:14)
N
i=1 1ait >(cid:14)

,

(3.1)

E ≡

where

Eit ≡ log10
(|Δit /cit |))
(cid:15)
−ρ
t+1

Δit ≡ ct −

β REt [c

(cid:16)− 1

ρ ,

]

and (cid:14) = 0.02. A value of E of − 2 and − 4 are interpreted as average approximation
errors of respectively 1 and 0.01% of consumption.

The results are shown in Table 1 for the C++ implementation of the proposed
solution methods. Starting with VFI in the ﬁrst column, we see that the model is
solved in about an hour, and that most of the time is spend on the keeper problem.
Continuing to NVFI in column two, we see that the model is now solved about 10
times faster with only a small reduction in accuracy due to the additional layers of
interpolation. The keeper problem is solved about 15 times faster than in VFI because
of the use of the post-decision value function, which it takes less than two minutes to
compute. The adjuster problem is also solved much faster due to both the use of the
post-decision value function and the reduction in the dimensionality of the decision
space. Turning to NEGM in column three, we see a further reduction in the time it
takes to solve the keeper problem, but some of the initial gain is lost by more time
spend in computing the post-decision functions, where the post-decision marginal

123

A Guide on Solving Non-convex Consumption-Saving Models

761

Table 1 Speed and accuracy

Relative Euler errors

All (average)

5th percentile

95th percentile

Adjusters (average)

Keepers (average)

Timings (in min, best of 5)

Total
Post− decision functions
Keeper problem

Adjuster problem
Speed− up relative to VFI
Simulation outcomes

Expected discounted utility
Adjuster share (dt (cid:5)= nt )
Average consumption (ct )
Variance of consumption (ct )
Average durable stock (dt )
Variance of durable stock (dt )

VFI

NVFI

NEGM

NVFI+

NEGM+

− 4.670
− 5.730
− 3.694
− 4.558
− 4.691

63.05
0.00
61.79
1.26

− 32.213
0.172
0.979
0.256
0.562
0.112

− 4.613
− 5.620
− 3.704
− 4.737
− 4.590

5.96
1.76
4.19
0.01
10.58

− 32.213
0.172
0.979
0.256
0.562
0.112

− 4.709
− 5.581
− 3.775
− 4.888
− 4.676

4.07
3.51
0.54
0.01
15.51

− 32.213
0.172
0.979
0.256
0.562
0.112

− 4.613
− 5.620
− 3.704
− 4.737
− 4.590

4.68
0.47
4.19
0.01
13.47

− 32.213
0.172
0.979
0.256
0.562
0.112

− 4.709
− 5.581
− 3.775
− 4.888
− 4.676

1.32
0.77
0.54
0.01
47.60

− 32.213
0.172
0.979
0.256
0.562
0.112

The model is solved backwards for T = 50 periods (iterations) using the implementation settings described
in Sect. 2.5. The simulation outcomes are computed from a sample of 100,000 households, and the Euler
errors are calculated as in Eq. (3.1). The optimization problems are solved by a simple golden section
search in both NVFI and NEGM. The optimization problems in VFI is solved by the method of moving
asymptotes from Svanberg (2002), implemented in NLopt by Johnson (2014), which was found to be the
fastest solver. The code was run using 8 threads on a Windows 10 computer with two Intel(R) Xeon(R) Gold
6154 3.00 GHz CPUs (18 cores, 36 logical processes each) and 192 GB of RAM. The code was compiled
with the free Microsoft Visual Studio 2017 C++ compiler with full optimization (Ox ﬂag) and parallelization
with OpenMP. The timing reported is best out of 3 runs

value of cash must now also be calculated. Compared to NVFI and VFI, NEGM also
provides an improvement in the level of accuracy. Turning to NVFI+ and NEGM+ we
see that the post-decision functions are computed more than three times faster, which
especially beneﬁts NEGM, where this step is the bottleneck. In the end, NEGM+ is
almost 50 times faster than VFI for the same accuracy. Across all solution methods
selected simulation outcomes (bottom part of Table 1), including expected discounted
utility, is indistinguishable from each other.

The speed-up factors presented above is naturally dependent on the chosen model
and its implementation. We have for instance completely side-stepped the issue of
using multi-start, when solving the keeper and adjuster problems. The non-concavity
of the value function implies that some form of multi-start is required to limit the risk
that the numerical solver ends up in a local maximum. Introducing multi-starts will
in particular increase solution time for VFI, where all the time is spend on solving
the keeper and adjuster problems. Solution time will also increase substantially for

123

762

J. Druedahl

NVFI, where most of the time is spend on the keeper problem. For NEGM, however,
the solution time will almost not increase because the keeper problem is solved with
EGM and the time spend on the adjuster problem is negligible. Adding multi-starts
would therefore increase the speed-up substantially.

Another important margin is the number of quadrature nodes required for numerical
integration. The computational costs of VFI is almost proportional to the number
of quadrature nodes. In NVFI and NEGM, the cost of computing the post-decision
functions is also proportional to the number of quadrature nodes, but the cost of solving
the keeper and adjuster problems are independent of the number of quadrature nodes.
With more (less) quadrature nodes the speed-ups relative to VFI will therefore increase
(decrease). Likewise the chosen density of the post-decision grid is important. For a
more dense grid the relative speed-up of NVFI and NEGM decrease, but their accuracy
increase.

The complexity of the adjuster problem can also be important. In a model with a
more complex adjuster problem in terms of more states and/or choices, there still will
be a beneﬁt of going from VFI to NVFI(+). If solving the adjuster problem becomes
the bottleneck in NVFI(+) the improvement of going to NEGM(+) will, however, be
low in relative terms.

Table 2 presents speed results for the Python implementations of NVFI+ and
NEGM+ optimized with just-in-time computation using the Numba package. Solution
time only increases with a factor of about 1.5 in the Python implementations of NVFI+
and NEGM+.13

3.1 Extended Model

In this sub-section, I consider an extended version of the benchmark model, where
t ) with different depreciation rates (δ1,
there are two different durable stocks (d1
1 , d2
δ2), different adjustment cost factors (τ1, τ2), and potentially differential impact on
utility,

u(ct , d1
t

, d2
t

) =

(cα
t

((d1
t

+ d1

+ d2

)γ (d2
t
1 − ρ

)1−γ )1−α)1−ρ

, α, γ ∈ (0, 1), ρ > 0.

This adds a state variable to the keeper problem, and imply that there are three different
adjuster problems depending on whether one or both of the durable stocks are adjusted.
The implied Bellman equations and further implementation details are provided in
“Appendix”.

Table 3 shows speed and accuracy results for VFI, NVFI+ and NEGM+. The speed-
up factor provided by NVFI+ increases to around 20, while the speed-up factor for
NEGM+ mildly decreases to 44. The accuracy NEGM+ of is now slightly worse than
VFI.

13 Comparing various approaches to parallelization on CPUs Fernandez-Villaverde and Valencia (2018)
found C++ with either OpenMP or MPI to be the fastest.

123

A Guide on Solving Non-convex Consumption-Saving Models

763

Table 2 Speed and Accuracy in Python

Relative euler errors

All (average)

5th percentile

95th percentile

Adjusters (average)

Keepers (average)

Timings (in min, best of 5)

Total

Post-decision functions

Keeper problem

Adjuster problem

Simulation outcomes

Expected discounted utility
Adjuster share (dt (cid:5)= nt )
Average consumption (ct )
Variance of consumption (ct )
Average durable stock (dt )
Variance of durable stock (dt )

NVFI+

C++

− 4.613
− 5.620
− 3.704
− 4.737
− 4.590

4.68
0.47
4.19
0.01

− 32.213
0.172
0.979
0.256
0.562
0.112

Python

− 4.613
− 5.620
− 3.704
− 4.737
− 4.590

7.07
0.92
6.12
0.02

− 32.213
0.172
0.979
0.256
0.562
0.112

NEGM+

C++

− 4.709
− 5.581
− 3.775
− 4.888
− 4.676

1.32
0.77
0.54
0.01

− 32.213
0.172
0.979
0.256
0.562
0.112

Python

− 4.709
− 5.581
− 3.775
− 4.888
− 4.676

2.29
1.60
0.67
0.02

− 32.213
0.172
0.979
0.256
0.562
0.112

See Table 1. The Python code is optimized with just-in-time compilation from the Numba package

4 When Can NEGM be Used?

In this section, NEGM is presented in more general terms. The assumptions made
below are sufﬁcient for using NEGM, though not always necessary.

4.1 Model Class

Let mt denote beginning-of-period cash-on-hand (i.e. liquid resources), ct denote
, . . . , n#N
consumption, and at denote end-of-period liquid assets. Further let nt = (n1
)
t
t
, . . . , d#D
) ∈ D(Mt , Nt ) a
be a vector of the #N other states than mt , and dt = (d1
t
t
, . . . , b#B
vector of the #D other choices than ct . Also let bt = (b1
) be a vector of the
t
t
remaining post-decision states besides at . Assume that bt for some function B is given
by

Bt = B(nt , dt ).

(4.1)

123

764

J. Druedahl

Table 3 Speed and accuracy with two durable stocks

VFI

NVFI+

NEGM+

Relative euler errors

All (average)

5th percentile

95th percentile

Adjusters (average)

Keepers (average)

Timings (in min)

Total

Post-decision functions

Keeper problem

Adjuster problem

Speed-up relative to VFI

Simulation outcomes

(cid:5)= n2
t )

∨ d2
t

(cid:5)= n1
t

Expected discounted utility
Adjuster share (d1
t
Average consumption (ct )
Variance of consumption (ct )
Average durable stock I (d1
t )
Variance of durable stock I (d1
t )
Average durable stock II (d2
t )
Variance of durable stock II (d2
t )

− 3.848
− 4.906
− 2.853
− 3.732
− 3.895

265.41
0.00
234.73
30.68

− 34.288
0.303
0.981
0.256
0.380
0.051
0.217
0.018

− 3.746
− 4.864
− 2.047
− 3.832
− 3.711

13.71
3.08
10.41
0.22
19.36

− 34.289
0.304
0.981
0.256
0.379
0.050
0.216
0.018

− 3.582
− 4.735
− 2.365
− 3.822
− 3.479

6.00
4.91
0.86
0.22
44.25

− 34.295
0.310
0.978
0.254
0.377
0.050
0.214
0.018

The model is solved backwards for T = 50 periods (iterations) using the implementation settings described
in “Appendix”. The simulation outcomes are computed from a sample of 100,000 households, and the
Euler errors are calculated as in Eq. (3.1). The optimization problems are solved by a simple golden section
search in both NVFI and NEGM. The optimization problems in VFI is solved by the method of moving
asymptotes from Svanberg (2002), implemented in NLopt by Johnson (2014), which was found to be the
fastest solver. The code was run using 8 threads on a Windows 10 computer with two Intel(R) Xeon(R)
Gold 6154 3.00 GHz CPUs (18 cores, 36 logical processes each) and 192 GB of RAM. The code was
compiled with the free Microsoft Visual Studio 2017 C++ compiler with full optimization (Ox ﬂag) and
parallelization with OpenMP

For a vector of stochastic shocks, ψt+1, let the transition functions for mt and nt be
denoted Tm and Tn. Assume, for simplicity, that Tm is continuous, differentiable and
strictly monotone wrt. at .14 Also assume that Tn is independent of at .15

The fundamental restrictive assumption so far, is that consumption, ct , only affects
the future through its effect on ﬁrst end-of-period assets, at , and then future cash-on-

14 If Tm is not differentiable everywhere in at there will be multiple Euler-equations. If there are k kinks
where Tm is not differentiable in at , there will be k + 1 intervals where Tm is differentiable in at . Conse-
quently there will be k + 1 mutually exclusive Euler-equations. Interior optimal consumption choices not
at kinks must still satisfy one of these, and the proposed NEGM can therefore still be applied.
15 If Tn is continuous and differentiable wrt. to at it will still be possible to derive an equation like the
standard Euler-equation. The fast vectorized interpolation scheme can, however, not be straightforwardly
used as changes in at then affect both mt+1 and nt+1.

123

A Guide on Solving Non-convex Consumption-Saving Models

765

hand, mt+1. This is a mild assumption satisﬁed in most models, but e.g. not models
with habit formation in consumption.

Further assume that the nt states and dt choices only affect end-of-period assets

additively, i.e. that we for some function L can write

lt = mt + L(nt , dt )
at = lt − ct ,

(4.2)
(4.3)

where lt are liquid assets just before consumption. This is also a rather weak assump-
tion, and fulﬁlled in most consumption-saving models in the literature. Likewise
assume that the minimum level of end-of-period liquid assets is only a function of
the other post-decision states, i.e. at ≥ a(bt ). This implies that the maximum level of
consumption is given by c(lt , bt ) ≡ lt − a(bt ).

Finally, assume that the effect of nt and dt on utility is additively separable such

that the period utility function for functions u and h has the form

u(ct , bt ) + h(nt , dt ),

(4.4)

where it is required that uc and ucc exists, uc > 0 and ucc < 0. The assumption of
additive separability for nt and dt can be loosed somewhat, but at a potential cost as
discussed below. This might e.g. be interesting in models with labor supply.

In sum, the Bellman equation for the considered model class can be written as

Vt (mt , nt ) = max
ct ,dt

u(ct , bt ) + h(nt , dt ) + βEt

(cid:17)
vt+1(mt+1, nt+1)

(cid:18)

s.t.

lt = mt + L(nt , dt )
at = lt − ct
bt = B(nt , dt )

mt+1 = TM (at , bt , ψt+1)
nt+1 = TN (bt , ψt+1)
dt ∈ D(mt , nt )
ct ∈ [0, c(lt , bt )],

(4.5)

where β is the discount factor. Denote the optimal choice functions by c(cid:17)
t
and d(cid:17)
(mt , nt ), and the implied optimal post-decision functions by a(cid:17)
t
t
b(cid:17)
(mt , nt ).
t
Using a variational argument, it can be shown that optimal interior consumption

(mt , nt )
(mt , nt ) and

choices, ct ∈ (0, c(•)), must satisfy an Euler-equation

uc(ct , bt ) = qt (at , bt ),

(4.6)

where

qt (at , bt ) = βEt [Tm,a,t+1uc,t+1].

123

766

4.2 NEGM

J. Druedahl

In order to reformulate the problem in Eq. (4.5), it is beneﬁcial to introduce the post-
decision value function given by

wt (at , bt ) ≡ βEt

(cid:17)
vt+1(TM (at , bt , ψt+1), TN (bt , ψt+1))

(cid:18)

.

(4.7)

Next, deﬁne the following pure consumption problem

Vt (lt , bt ) = max

ct

u(ct , bt ) + wt (at , bt )

s.t.

at = lt − ct
ct ∈ [0, c(lt , bt )],

(4.8)

If the separability in Eq. (4.4) was not assumed, then dt should for example also be
a state in the pure consumption problem. In labor supply models with human capital
accumulation, for example, the pure consumption problem might depend not just of
end-of-period human capital, but also on current labor supply if consumption and
leisure are non-separable.

Solving the pure consumption problem is just like solving the keeper problem in
the benchmark model. The z(at , bt ) function is found as the solution for ct of Eq.
(4.6). And then lt is found by lt = at + ct as implied by Eq. (4.3). The upper envelope
is applied just like in the benchmark model.

Based on the pure consumption problem, the model in Eq. (4.5) can be reformulated

as,

vt (mt , nt ) = max
dt

Vt (lt , bt ) + h(nt , dt )

s.t.

bt = B(nt , dt )
lt = mt + L(nt , dt )
dt ∈ D(mt , nt ).

(4.9)

This reformulation is greatly beneﬁcial for two reasons. Firstly, it reduces the dimen-
sionality of the optimization problem (only max over dt , not also ct ). Secondly, it
replaces multiple interpolations of the next period value function required to calculate
the continuation value, Et [Vt+1(•)], with interpolation of just Vt (lt , bt ). As in the
benchmark model, the suggestion is to solve this model with standard value function
iteration.

5 Comparison with G2EGM

The benchmark model used in this paper cannot be solved by the G2EGM proposed in
Druedahl and Jørgensen (2017). The basic idea in G2EGM is to ﬁx the post-decision

123

A Guide on Solving Non-convex Consumption-Saving Models

767

decision states (in the benchmark model this is pt , dt and at ) and then invert a system
of ﬁrst order conditions and post-decision state equations to ﬁnd choices and pre-
decision states. In the benchmark model, focusing on the adjuster problem, the relevant
equations can be written as

uc(ct , dt ) = wt,a( pt , dt , at )
ud (ct , dt ) = wt,a( pt , dt , at ) − wt,d ( pt , dt , at )

xt = at + ct + dt .

The problem is that this equation system is not invertible in the sense required by
G2EGM. Consider ﬁxing the post-decision states pt , dt and at . The ﬁrst equation can
then be used to ﬁnd the consumption choice, ct , and the third equation can be used to
ﬁnd the cash-on-hand state, xt . But when pt , dt , at and ct are all determined, both the
LHS and the RHS of the second equation is ﬁxed. The problem is that nothing insures
it is satisﬁed. Therefore G2EGM cannot be applied for the benchmark model.

Instead, I present a version of the model originally used Druedahl and Jørgensen

(2017) and solve it with both G2EGM and NEGM.

5.1 Model

Households are either retired or working. Working households can retire. Retired
households can not begin to work again. In retirement, households solve a standard
consumption-savings problem. The resources available for consumption in period t is
denoted mt , such that the post-decision assets at is given by

at = mt − ct .

Next period resources are given by

mt+1 = Raat + y,

where Ra is the return factor, and y is a (deterministic) retirement income. We assume
that households are not allowed to borrow, at ≥ 0.

Working households solve a more general problem, and are allowed to save in both
liquid assets (at ) and illiquid pension assets (bt ). Denoting the pension fund deposits
by dt , we assume that the post-decision (or end-of-period) assets levels are given by

at = mt − ct − dt
bt = nt + dt + g(dt ),

where g(dt ) is a pension deposit function potentially allowing for an extra incentive
to accumulate illiquid pension funds due to, for example, tax deductions of pension
contributions. Deposits are required to be non-negative, i.e. dt ≥ 0, and the assumption
of no borrowing, at ≥ 0, is also maintained for the working households.

123

768

J. Druedahl

The resources available for consumption and pension savings in the next period are

mt+1 = Raat + ηt+1, log ηt+1 ∼ N (−0.5σ 2
nt+1 = Rbbt

η , σ 2
η )

where ηt is stochastic labor income, and we assume a higher return on pension assets
than on liquid assets, i.e. Rb ≥ Ra.

Denoting the discrete choice of retirement by zt = 0 and the discrete choice of

working by zt = 1, the Bellman equation of the model can be formulated as

Vt (zt−1, mt , nt ) =

max
zt ∈Zt (zt−1)

vt (0, mt + nt )
vt (1, mt , nt )

if zt = 0
if zt = 1

(cid:2)

s.t.

(cid:2)

Zt (zt−1) =

{0, 1}
0

if zt−1 = 1
if zt−1 = 0

The discrete-choice-speciﬁc value function for the working households is

vt (1, mt , nt ) = max
ct ,dt

u(ct ) − α + βEt

(cid:17)

Vt+1(1, mt+1, nt+1)

(cid:18)

s.t.

at = mt − ct − dt
bt = nt + dt + g(dt )

mt+1 = Raat + ηt+1
nt+1 = Rbbt
ct ≥ 0
dt ≥ 0
ct + dt ∈ [0, mt ],

(5.1)

(5.2)

where u(ct ) denotes per-period utility ﬂow from consuming, ct , and α is the disu-
tility of labor. The discrete-choice-speciﬁc value function for the retiring (or retired)
households is

vt (0, xt ) = max

ct

u(ct ) + βVt+1(0, mt+1, 0)

s.t.

at = xt − ct
mt+1 = Raat + y
ct ∈ [0, xt ].

(5.3)

123

A Guide on Solving Non-convex Consumption-Saving Models

We assume the following functional forms

u(ct , zt ) = c1−ρ
t
1 − ρ
g(dt ) = χ log(1 + dt ).

769

(5.4)

(5.5)

5.2 Solution

The problem of the retired households can be solved with standard EGM, and it
is independent of the problem of the working households because retirement is an
absorbing state. Further details are provided in Druedahl and Jørgensen (2017).

In G2EGM the problem of the working households is solved by ﬁrstly computing
the post-decision value function and it’s derivatives, and then solving the full problem
with EGM using ﬁrst order conditions for both the consumption choice and the pension
deposit choice. Details are again provided in Druedahl and Jørgensen (2017).

For NEGM the post-decision value is again computed ﬁrst, but now only the deriva-
tive wrt. to end-of-period assets is needed. The problem of the working households is,
however, solved in two separate steps. First the pure consumption problem is solved.
Then the optimal deposit choice is determined.
Deﬁning the post decision value function

wt (at , bt ) = βEt [Vt+1(1, mt+1, nt+1)],

the pure consumption problem is

Vt (1, lt , bt ) = max

ct

u(ct ) − α + wt (at , bt )

s.t.

at = lt − ct
mt+1 = Raat + ηt+1
nt+1 = Rbbt

ct ∈ [0, lt ].

(5.6)

This can be solved with EGM as explained in the previous section. The optimal deposit
choice is then determined in an outer problem by

vt (1, mt , nt ) = max
dt

Vt (1, lt , bt )

s.t.

lt = mt − dt
bt = nt + dt + g(dt )
dt ∈ [0, mt ].

(5.7)

123

770

J. Druedahl

Table 4 Comparing G2EGM and NEGM

Relative Euler errors

All (average)

5th percentile

95th percentile

Timings (in min, best of 5)

Total
Post− decision functions
EGM− step
VFI− step

σ 2
η = 0.0
G2EGM

− 6.233
− 7.361
− 4.266

0.81
0.03
0.78

NEGM

− 5.367
− 7.205
− 3.347

1.08
0.03
0.33
0.72

σ 2
η = 0.1
G2EGM

− 5.493
− 6.768
− 3.964

1.22
0.44
0.78

NEGM

− 5.208
− 6.588
− 3.708

1.36
0.32
0.31
0.73

In the speciﬁcation with income risk, σ 2
η = 0.1, 16 quadrature nodes is used. Otherwise the exact same
parameters and grids as in Druedahl and Jørgensen (2017) were used. The optimization problem in NEGM
is solved by a simple golden section search. The code was run using 1 thread on a Windows 10 computer
with two Intel(R) Xeon(R) Gold 6154 3.00 GHz CPUs (18 cores, 36 logical processes each) and 192 GB
of RAM. The Python code is optimized with just-in-time compilation from the Numba package. A serial
implementation is used because the two-dimensional upper envelope of G2EGM is not straightforward to
parallellize in Python when there is no exogenous states. Only the time spend solving the working household
problem is included

5.3 Speed and Accuracy

Table 4 shows speed and accuracy results for G2EGM and NEGM using Python imple-
mentations.16 I consider one speciﬁcation without income risk, and one speciﬁcation
with income risk and 16 quadrature nodes. Otherwise, I use the exact same parameters
and grid sizes as in Druedahl and Jørgensen (2017) with the number of nodes in the
grid for mt at #m = 600, and calculate Euler errors as they do.

For the pure consumption problem in NEGM, I use the post-decision grid for bt
and for lt I use the same grid as for mt . The outer VFI step in NEGM is solved
with a golden section search algorithm extended with explicit checks for whether a
constrained choice is superior. For both G2EGM and NEGM, the fast interpolation
approach developed in this paper is used.

In the speciﬁcation without income risk G2EGM provides a speed-up of about 25%
relative to NEGM, and the accuracy of G2EGM is superior. In the speciﬁcation with
income risk the speed-up is only about 10%, and the accuracy of G2EGM is only
slightly better. The result changes because the post-decision functions becomes more
costly to compute in the presence of income shocks due to numerical integration.
NEGM is furthermore less demanding in this regarding because fewer derivatives are
used.

If there were additional shocks, or more quadrature nodes were used, the results
would likely turn in favor of NEGM. On the other hand, it could be argued that the VFI

16 A serial implementation is used because the two dimensional upper envelope of G2EGM is not straight-
forward to parallellize in Python when there is no exogenous states.

123

A Guide on Solving Non-convex Consumption-Saving Models

771

step in NEGM should use a multi-start algorithm to limit the risk of convergence to
local maxima. Models with a more costly outer problem, where VFI is used in NEGM,
would also beneﬁt G2EGM. Adjusting the relative size of grids also affect the results.
In sum, however, these results show that even in models where both G2EGM and
NEGM can be used, NEGM might be an OK choice purely in terms of speed. An exam-
ination of the code base furthermore reveal that NEGM is much easier to implement
than G2EGM, and as explained previously is applicable to a wider set of models.

6 Conclusions

We have seen that the solution methods proposed in this paper, the Nested Value Func-
tion Iteration (NVFI) and the Nested Endogenous Grid Method (NEGM), provide
substantial speed-up compared to standard value function iteration (VFI). Further-
more, we have seen that the methods are applicable to a large class of models, and
straightforward to implement as they rely on generic algorithms for interpolation and
for ﬁnding the upper envelope across candidate solutions to the Euler-equation.

Faster solution methods makes it possible to solve and estimate richer consumption-
saving models than previously, and thus makes it computationally feasible to perform
policy analysis based on more realistic models.

Acknowledgements I thank Anders Munk-Nielsen, Bertel Schjerning, Alessandro Martinello, Giulio Fella,
Matthew White and especially Thomas Høgholm Jørgensen for fruitful discussions and suggestions. The
usual disclaimer applies. An earlier version of this paper has been circulated under the title “A fast nested
endogenous grid method for solving general consumption-saving models”. Financial support from the
Danish Council for Independent Research in Social Sciences (Grant No. 5052-00086) is gratefully acknowl-
edged.

Appendix: Model in Sect. 3.1

Bellman Equation

Deﬁning the vector of state variables by st = ( pt , n1
, mt ). The recursive Bellman
t
equation for the extended benchmark model with two durable stocks can be written

, n2
t

vt (st ) = max

⎧
⎪⎪⎪⎨
⎪⎪⎪⎩

, mt )

vkeep
t

, n2
( pt , n1
t
t
vad j.
( pt , xt )
t
vad j.,1
( pt , n2
t
t
vad j.,2
( pt , n2
t
t

, x 1
t
, x 2
t

)
)

⎫
⎪⎪⎪⎬
⎪⎪⎪⎭

s.t.

xt = mt + (1 − τ1)n1
t
= mt + (1 − τ1)n1
x 1
t
t
= mt + (1 − τ2)n2
x 2
t
t

+ (1 − τ2)n2
t
,

,

(6.1)

123

vkeep
t

( pt , n1
t

, n2
t

, mt ) = max

ct

u(ct , n1
t

, n2
t

) + βEt [vt+1(st+1)]

s.t.

at = mt − ct

mt+1 = (1 + r )at + yt+1
= (1 − δ1)n1
n1
t+1
t
= (1 − δ2)n2
n2
t+1
t
at ≥ 0,

vad j.
t

( pt , xt ) = max
,d2
ct ,d1
t
t

s.t.

u(ct , d1
t

, n2
t

) + βEt [vt+1(st+1)]

− d2
at = xt − ct − d1
t
t
mt+1 = (1 + r )at + yt+1
= (1 − δ1)d1
n1
t+1
t
= (1 − δ2)d2
n2
t+1
t
at ≥ 0,

vad j.,1
t

( pt , n2
1

, x 1
t

) = max
ct ,d1
t

u(ct , d1
t

, n2
t

) + βEt [vt+1(st+1)]

s.t.
at = x 1
t

− ct − d1
t

mt+1 = (1 + r )at + yt+1
= (1 − δ1)d1
n1
t+1
t
= (1 − δ2)n2
n2
t+1
t
at ≥ 0,

vad j.,2
t

( pt , n1
1

, x 2
t

) = max
ct ,d2
t

u(ct , n1
t

, d2
t

) + βEt [vt+1(st+1)]

s.t.
at = x 2
t

− ct − d2
t

mt+1 = (1 + r )at + yt+1
= (1 − δ1)n1
n1
t+1
t
= (1 − δ2)d2
n2
t+1
t
at ≥ 0.

J. Druedahl

(6.2)

(6.3)

(6.4)

(6.5)

772

where

and

and

and

123

A Guide on Solving Non-convex Consumption-Saving Models

773

Nesting

Deﬁning the post-decision value function

wt ( pt , d1
t

, d2
t

, at ) = βEt [vt+1( pt+1, n1

t+1

, n1

t+1

, mt+1)],

the keeper and adjuster value functions can be re-formulated as

vkeep
t

( pt , n1
t

, n2
t

, mt ) = max

ct

u(ct , n1
t

, n2
t

) + wt ( pt , d1
t

, d2
t

, at )

and

and

and

s.t.

at = mt − ct

mt+1 = (1 + r )at + yt+1
= (1 − δ1)n1
n1
t+1
t
= (1 − δ1)n2
n2
t+1
t
at ≥ 0,

vad j.
t

( pt , xt ) = max
,d2
t

d1
t

vkeep
t

( pt , d1
t

, d2
t

, mt )

s.t.

mt = xt − d1
t
∈ [0, xt ],

+ d2
t

d1
t

− d2
t

vad j.,1
t

( pt , n2
1

, x 1
t

) = max
,

d1
t

vkeep
t

( pt , d1
t

, n2
t

, mt )

(6.6)

(6.7)

s.t.
mt = x 1
t
∈ [0, x 1
d1
t
t

− mt − d1
t

],

(6.8)

vad j.,2
t

( pt , n1
1

, x 2
t

) = max
d2
t

vkeep
t

( pt , n1
t

, d2
t

, mt )

s.t.
mt = x 2
t
∈ [0, x 2
d2
t
t

− d2
t
].

(6.9)

123

774

EGM

J. Druedahl

Deﬁne the post-decision marginal value of cash by

q( pt , dt , at ) = β REt

(cid:5)
αc

α(1−ρ)−1
t+1

((d1

t+1

+ d1)γ (d2

t+1

+ d2)1−γ )(1−α)(1−ρ)

(cid:6)

.

(6.10)

The Euler-equation for ct is then given by

uc(ct , d1
t

, d2
t

) = q( pt , d1
t

, d2
t

, at ).

(6.11)

This implies that EGM can be performed as follows

ct = z(at , d1
t

, d2
t

, pt )

(6.12)

(cid:3)

=

α((d1
t
mt = at + ct .

+ d1

qt ( pt , dt , at )
)γ (d2
+ d2
t

)1−γ )(1−α)(1−ρ)

(cid:4) 1

α(1−ρ)−1

(6.13)

The upper envelope algorithm is unchanged.

Implementation

=
= 10−2, R = 1.03, τ1 = 0.08, τ1 = 0.12, δ1 = 0.10, δ2 = 0.20, γ = 0.5,

Parametrization The chosen parameters are β = 0.965, ρ = 2, α = 0.9 d1
d2
σψ = σξ = 0.1, and λ = 1.
Grids Constructed as in the benchmark model, but with the following changes # p =
50, n = 2, #n = 50, m = 10, #m = 100, x = 12, #x = 100, #a = 100.
Interpolation Done as in the benchmark model.

Simulation The initial values in the simulation are drawn as:

log p0 ∼ N (log(1), 0.2)
log d1
∼ N (log(0.4), 0.2)
0
log d1
∼ N (log(0.4), 0.2)
0
log a0 ∼ N (log(0.2), 0.1).

123

A Guide on Solving Non-convex Consumption-Saving Models

775

References

Barillas, F., & Fernández-Villaverde, J. (2007). A generalization of the endogenous grid method. Journal

of Economic Dynamics and Control, 31(8), 2698–2712.

Berger, D., & Vavra, J. (2015). Consumption dynamics during recessions. Econometrica, 83(1), 101–154.
Bertsekas, D. P. (2012) Dynamic programming and optimal control: Approximate dynamic programming.

Athena Scientiﬁc

Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization

problems. Economics Letters, 91(3), 312–320.

Druedahl, J., & Jørgensen, T. H. (2017). A general endogenous grid method for multi-dimensional models
with non-convexities and constraints. Journal of Economic Dynamics and Control, 74, 87–107.
Fella, G. (2014). A generalized endogenous grid method for non-smooth and non-concave problems. Review

of Economic Dynamics, 17(2), 329–344.

Fernandez-Villaverde, J., & Valencia, D. Z. (2018). A practical guide to parallization in economics. Tech-

nical report.

Harmenberg, K., & Oberg, E. (2017). Consumption dynamics under time-varying unemployment risk.

Working Paper.

Hintermaier, T., & Koeniger, W. (2010). The method of endogenous gridpoints with occasionally binding
constraints among endogenous variables. Journal of Economic Dynamics and Control, 34(10), 2074–
2088.

Hull, I. (2015). Approximate dynamic programming with post-decision states as a solution method for

dynamic economic models. Journal of Economic Dynamics and Control, 55, 57–70.

Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017). The endogenous grid method for discrete-
continuous dynamic choice models with (or without) taste shocks. Quantitative Economics, 8(2),
317–365.

Johnson, S. G. (2014). The NLopt nonlinear-optimization package
Jørgensen, T. H. (2013). Structural estimation of continuous choice models: Evaluating the EGM and MPEC.

Economics Letters, 119(3), 287–290.

Judd, K. L. (1992). Projection methods for solving aggregate growth models. Journal of Economic Theory,

58(2), 410–452.

Judd, K. L., Maliar, L., & Maliar, S. (2017). How to solve dynamic stochastic models computing expectations

just once. Quantitative Economics, 8(3), 851–893.

Low, H., & Meghir, C. (2017). The use of structural models in econometrics. Journal of Economic Perspec-

tives, 31(2), 33–58.

Ludwig, A., & Schön, M. (2018). Endogenous grids in higher dimensions: Delaunay interpolation and

hybrid methods. Computational Economics, 51, 1–30.

Ma, Q., & Stachurski, J. (2018). Dynamic programming deconstructed. Technical report.
Powell, W. B. (2011). Approximate dynamic programming: Solving the curses of dimensionality. New York:

Wiley.

Rendahl, P. (2015). Inequality constraints and Euler equation-based solution methods. The Economic Jour-

nal, 125(585), 1110–1135.

Santos, M. S. (2000). Accuracy of numerical solutions using the Euler equation residuals. Econometrica,

68(6), 1377–1402.

Svanberg, K. (2002). A class of globally convergent optimization methods based on conservative convex

separable approximations. SIAM Journal on Optimization, 12(2), 555–573.

Van Roy, B., Bertsekas, D.P., Lee, Y., & Tsitsiklis, J.N. (1997). A neuro-dynamic programming approach
to retailer inventory management. In Decision and control, 1997., Proceedings of the 36th IEEE
conference on (Vol. 4, pp. 4052–4057). IEEE.

White, M. N. (2015). The method of endogenous gridpoints in theory and practice. Journal of Economic

Dynamics and Control, 60, 26–41.

Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps
and institutional afﬁliations.

123


