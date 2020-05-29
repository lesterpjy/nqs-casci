// The following is part of the source code for the NQS project
// published on JCTC (dx.doi.org/10.1021/acs.jctc.9b01132) titled
// "Artifical Neural Networks Applied as Molecular Wave Function Solvers".
//
// It relies on the ORZ quantum chemistry computing package detailed in
// Yanai et al. (doi.org/10.1002/qua.24808). It is purely excerpted for
// demostration purposes, and cannot run as a stand-alone program.
//
// Copyright (C) Takeshi Yanai (yanait@gmail.com)
// Yang Peng−Jian (pjy@berkeley.edu)
// All Rights Reserved.
//
// ...... //
// −− [deterministic/metropolis] −−
inline void updateParam(const double momentum, const size_t iter, const long N,
                        const long M, orz::DTensor &vec_0, orz::DTensor &vec_1,
                        const orz::DTensor &vec, const double scaling,
                        orz::DTensor &ratio_0, orz::DTensor &ratio_1,
                        orz::DTensor &velocity_0, orz::DTensor &velocity_1) {
  //
  // Function: Update NN params
  // Description: The fucntion update NN bias and weight params
  // and prints update/param ratio to monitor learning
  // Toggle to momentum update available
  // Parameters: param matrices, update matrix, ratio matrix
  // Returns: None
  // Effect: Modifes param matrices
  //
  // store value in temp
  orz::DTensor temp_0(N);
  orz::DTensor temp_1(N, N);
  for (size_t i = 0; i < N; ++i)
    temp_0.cptr()[i] = scaling * vec.cptr()[i];
  for (size_t i = 0; i < N * N; ++i)
    temp_1.cptr()[i] = scaling * vec.cptr()[N + i];
  // monitoring learning with update/param ratio
  cout << "update/bias ratio" << endl;
  for (size_t i = 0; i < N; ++i)
    ratio_0.cptr()[i] = std::abs(temp_0.cptr()[i] / vec_0.cptr()[i]);
  cout << "bias: \n" << ratio_0 << endl;
  cout << "norm: " << orz::tensor::normf(ratio_0) << endl;
  cout << "update/weight ratio; ideal = 1e−03" << endl;
  for (size_t i = 0; i < N * N; ++i)
    ratio_1.cptr()[i] = std::abs(temp_1.cptr()[i] / vec_1.cptr()[i]);
  cout << "weights: \n" << ratio_1 << endl;
  cout << "norm: " << orz::tensor::normf(ratio_1) << endl;
  // Update
  for (size_t i = 0; i < N; ++i)
    vec_0.cptr()[i]− = temp_0.cptr()[i];
  for (size_t i = 0; i < N * N; ++i)
    vec_1.cptr()[i]− = temp_1.cptr()[i];
  /*
  // Momentum update
  if (iter != 0){
  // parameter update
  for(size_t i = 0; i < N ; ++i) vec_0.cptr()[i] = momentum *
  velocity_0.cptr()[i] − temp_0.cptr()[i]; for(size_t i = 0; i < M ; ++i)
  vec_1.cptr()[i] = momentum * velocity_1.cptr()[i] − temp_1.cptr()[i];
  }
  else{
  // storing first iteration velocity
  for(size_t i = 0; i < N ; ++i) velocity_0.cptr()[i] = temp_0.cptr()[i];
  for(size_t i = 0; i < M ; ++i) velocity_1.cptr()[i] = temp_1.cptr()[i];
  // parameter update
  for(size_t i = 0; i < N ; ++i) vec_0.cptr()[i] −= velocity_0.cptr()[i];
  for(size_t i = 0; i < M ; ++i) vec_1.cptr()[i] −= velocity_1.cptr()[i];
  }
  */
}
// ...... //
inline void
evalgrad_metropolis(const size_t niters,
                    const size_t nDets, // <= N!
                    const double base_energy,
                    const std::vector<orz::fci::DetBits> &bitsvec, // <= N!
                    const orz::DTensor &bvec, const orz::DTensor &wmat,
                    const orz::DTensor &cvec, const orz::DTensor &xmat,
                    const size_t nw, const size_t N, const size_t Ma,
                    const size_t Mp, const orz::DTensor &matH,
                    const bool update_theta, std::vector<OptData> &datvec,
                    OptData &dat_reduced) // <<= N!

{
  //
  // Input: required as in function call
  // Output: None
  // Consequence: MH sampled distribution of configurations
  //
  //
  // Function: MH algorithm sampling/ calculates energy, gradients
  // Description: Utilize MH algorithm to generate random samples,
  // calculates energy, gradients within sampling
  // Parameters: As listed
  // Returns: None
  // Effect: Modify matrix to store update gradients
  //
  //
  const int nproc = orz::world().size();
  const int myrank = orz::world().rank();
  int naccept = 0;
  // data structures for evaluating Cv and Eloc
  // where cbar_v = cbar * sqrt(partition func)
  const auto nthread = orz::openmp::nthread();
  orz::ProgressTimer pt("time: evalgrad_metropolis:");
  {
    orz::ProgressTimer pt("time: thread part:");
#pragma omp parallel {
    const auto mythread = omp_get_thread_num();
    const auto dist_niters = ndistributed(niters, nthread, mythread);
    // cout << "niters=" << niters << " mythread=" << mythread << " " <<
    // dist_niters << " nw=" << nw << endl;
    auto &dat = datvec[mythread];
    dat.nsamples = 0 uL;
    for (size_t i = 0; i < nDets; ++i)
      dat.nsamples_of_det.cptr()[i] = 0 uL;
    dat.etot = 0.0; // sum of local energy
    dat.Ot_a <<= 0.0;
    dat.Ot_b <<= 0.0;
    dat.Ot_w <<= 0.0;
    dat.Ot_c <<= 0.0;
    dat.Ot_d <<= 0.0;
    dat.Ot_x <<= 0.0;
    dat.eloc_oa <<= 0.0;
    dat.eloc_ob <<= 0.0;
    dat.eloc_ow <<= 0.0;
    dat.eloc_oc <<= 0.0;
    dat.eloc_od <<= 0.0;
    dat.eloc_ox <<= 0.0;
    dat.S_the <<= 0.0;
    dat.S_tau <<= 0.0;
    double expmax = 0.0, expmax1 = 0.0, expmax2 = 0.0;
    // generate nw walkers of random, once
    for (size_t m = 0; m < nw; ++m)
      dat.walkers_curr[m] = dat.rand_ndets(dat.rand_gen);
    for (size_t i = 0; i < dist_niters; ++i) {
      // generate a random step, distributed uniformly
      for (size_t m = 0; m < nw; ++m)
        dat.walkers_step[m] = dat.rand_ndets(dat.rand_gen);
      // convert generated walkers into bitsvec
      for (size_t m = 0; m < nw; ++m)
        dat.walkers_curr_bits[m] = bitsvec[dat.walkers_curr[m]];
      for (size_t m = 0; m < nw; ++m)
        dat.walkers_step_bits[m] = bitsvec[dat.walkers_step[m]];
      // evaluate psi(v|theta) with bitsets of current and random step
      eval_enefunc(dat.walkers_curr_bits, bvec, wmat, nw, N, Ma,
                   dat.ene_curr_the);
      eval_enefunc(dat.walkers_step_bits, bvec, wmat, nw, N, Ma,
                   dat.ene_step_the);
      // calculate key prob:
      for (size_t m = 0; m < nw; ++m)
        dat.key_prob[m] = std::min(
            1.0,
            std::exp(dat.ene_step_the.cptr()[m]− dat.ene_curr_the.cptr()[m]));
      // generate uniform probability
      for (size_t m = 0; m < nw; ++m)
        dat.uni_prob[m] = dat.rand_unity(dat.rand_gen);
      // evaluate psi(v|tau) with to be accepted config(walkers_step_bits)
      eval_enefunc(dat.walkers_curr_bits, cvec, xmat, nw, N, Mp,
                   dat.ene_curr_tau);
      eval_enefunc(dat.walkers_step_bits, cvec, xmat, nw, N, Mp,
                   dat.ene_step_tau);
      // master equation: set walkers_curr = walkers_step with uniform prob
      for (size_t m = 0; m < nw; ++m) {
        if (dat.uni_prob[m] < dat.key_prob[m]) {
          naccept += 1;
          dat.walkers_curr[m] = dat.walkers_step[m];
          dat.walkers_curr_bits[m] = dat.walkers_step_bits[m];
          dat.ene_curr_the.cptr()[m] = dat.ene_step_the.cptr()[m];
          dat.ene_curr_tau.cptr()[m] = dat.ene_step_tau.cptr()[m];
        }
        // Here a configuration, v, has been sampled!!
        // cout << "walkers_curr: " << walkers_curr[m] << "(" <<
        // walkers_curr_bits[m] <<")" << endl; cout << "N=0: " <<
        // walkers_curr_bits[m][0] << endl;
        dcomplex eloc_mu = 0.0; // initialize Eloc (local energy)
        {
          // calculate cbar_mu
          ////// const auto cbar_mu = eval_cbar(dat.ene_curr_the.cptr()[m],
          /// dat.ene_curr_tau.cptr()[m]);
          const double ene_mu_the = dat.ene_curr_the.cptr()[m];
          const double ene_mu_tau = dat.ene_curr_tau.cptr()[m];
          // calculate sum of local energy for all <v|H|mu>
          for (size_t n = 0; n < nDets; ++n) {
            // considering the sparsity of Hamiltonian
            if (std::abs(matH.cptr()[dat.walkers_curr[m] * nDets + n]) >
                1 e− 12) {
              // n −> bitsvec[n]
              const auto ene_n_the =
                  eval_enefunc(bitsvec[n], bvec, wmat, N, Ma);
              const auto ene_n_tau =
                  eval_enefunc(bitsvec[n], cvec, xmat, N, Mp);
              const auto cbar_n_per_cbar_mu = eval_cbar1_per_cbar2(
                  ene_n_the, ene_n_tau, ene_mu_the, ene_mu_tau);
              eloc_mu += matH.cptr()[dat.walkers_curr[m] * nDets + n] *
                         cbar_n_per_cbar_mu;
              if (expmax < (ene_n_the− ene_mu_the)) {
                expmax = (ene_n_the− ene_mu_the);
                expmax1 = ene_n_the;
                expmax2 = ene_mu_the;
              }
            }
          }
        }
        // SLLN after warmup
        if (i >= (500 / nthread)) {
          dat.etot += eloc_mu;
          dat.nsamples += 1;
          dat.nsamples_of_det.cptr()[dat.walkers_curr[m]] += 1;
        }
        //−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
        // calculate partial derivatives(gradients) for updating theta, tau
        const orz::fci::DetBits &sampled_bits = dat.walkers_curr_bits[m];
        // for theta
        if (update_theta) {
          for (size_t j = 0; j < N; ++j) {
            const int vj = sampled_bits[j];
            dat.Ov_b.cptr()[j] = 0.5 * vj;
            dat.Ot_b.cptr()[j] += dat.Ov_b.cptr()[j];
            dat.eloc_ob.cptr()[j] += std::conj(eloc_mu) * dat.Ov_b.cptr()[j];
          }
          for (size_t i = 0; i < N; ++i) {
            // const double ov = eval_Ov(i, N, sampled_bits, bvec, wmat);
            const int vj_ = sampled_bits[i];
            for (size_t j = 0; j < N; ++j) {
              const int vj = sampled_bits[j];
              const size_t indx = i * N + j;
              dat.Ov_w.cptr()[indx] = vj * vj_;
              dat.Ot_w.cptr()[indx] += dat.Ov_w.cptr()[indx];
              dat.eloc_ow.cptr()[indx] +=
                  std::conj(eloc_mu) * dat.Ov_w.cptr()[indx];
            }
          }
          // gather Ov*Ovprime sum for calculating <ovovPrime> later
          eval_S(N, Ma, dat.Ov_b, dat.Ov_w, dat.size_the, +1.0, dat.S_the,
                 dat.workspc);
        }
        // for tau
        for (size_t j = 0; j < N; ++j) {
          const int vj = sampled_bits[j];
          dat.Ov_c.cptr()[j] = 0.5 * vj;
          dat.Ot_c.cptr()[j] += dat.Ov_c.cptr()[j];
          dat.eloc_oc.cptr()[j] +=
              std::conj(eloc_mu) * dcomplex(0.0, dat.Ov_c.cptr()[j]);
        }
        for (size_t i = 0; i < N; ++i) {
          // const double ov = eval_Ov(i, N, sampled_bits, dvec, xmat);
          const int vj_ = sampled_bits[i];
          for (size_t j = 0; j < N; ++j) {
            const int vj = sampled_bits[j];
            const size_t indx = i * N + j;
            dat.Ov_x.cptr()[indx] = vj * vj_;
            dat.Ot_x.cptr()[indx] += dat.Ov_x.cptr()[indx];
            dat.eloc_ox.cptr()[indx] +=
                std::conj(eloc_mu) * dcomplex(0.0, dat.Ov_x.cptr()[indx]);
          }
        }
        // gather Ov*Ovprime sum for calculating <ovovPrime> later
        eval_S(N, Mp, dat.Ov_c, dat.Ov_x, dat.size_tau, +1.0, dat.S_tau,
               dat.workspc);
      }
    } // end of iters
    if (mythread == 0)
      cout << "expmax = " << expmax << " " << expmax1 << " " << expmax2 << endl;
  } // end of thread
}
// reduce threads
for (long i = 1; i < datvec.size(); ++i) {
  for (long j = 0; j < datvec[0].nsamples_of_det.size(); ++j)
    datvec[0].nsamples_of_det.cptr()[j] += datvec[i].nsamples_of_det.cptr()[j];
  datvec[0].nsamples += datvec[i].nsamples;
  datvec[0].etot += datvec[i].etot;
  datvec[0].Ot_b += datvec[i].Ot_b;
  datvec[0].Ot_w += datvec[i].Ot_w;
  datvec[0].Ot_c += datvec[i].Ot_c;
  datvec[0].Ot_x += datvec[i].Ot_x;
  datvec[0].eloc_ob += datvec[i].eloc_ob;
  datvec[0].eloc_ow += datvec[i].eloc_ow;
  datvec[0].eloc_oc += datvec[i].eloc_oc;
  datvec[0].eloc_ox += datvec[i].eloc_ox;
  datvec[0].S_the += datvec[i].S_the;
  datvec[0].S_tau += datvec[i].S_tau;
}
// All reduce for all processes
dat_reduced.init(nDets, nw, N, Ma, Mp, 0);
boost::mpi::all_reduce(orz::world(), datvec[0].nsamples_of_det.cptr(),
                       datvec[0].nsamples_of_det.size(),
                       dat_reduced.nsamples_of_det.cptr(), std::plus<long>());
boost::mpi::all_reduce(orz::world(), datvec[0].nsamples, dat_reduced.nsamples,
                       std::plus<size_t>());
boost::mpi::all_reduce(orz::world(), datvec[0].etot, dat_reduced.etot,
                       std::plus<dcomplex>());
boost::mpi::all_reduce(orz::world(), datvec[0].Ot_b.cptr(),
                       datvec[0].Ot_b.size(), dat_reduced.Ot_b.cptr(),
                       std::plus<double>());
boost::mpi::all_reduce(orz::world(), datvec[0].Ot_w.cptr(),
                       datvec[0].Ot_w.size(), dat_reduced.Ot_w.cptr(),
                       std::plus<double>());
boost::mpi::all_reduce(orz::world(), datvec[0].Ot_c.cptr(),
                       datvec[0].Ot_c.size(), dat_reduced.Ot_c.cptr(),
                       std::plus<double>());
boost::mpi::all_reduce(orz::world(), datvec[0].Ot_x.cptr(),
                       datvec[0].Ot_x.size(), dat_reduced.Ot_x.cptr(),
                       std::plus<double>());
boost::mpi::all_reduce(orz::world(), datvec[0].eloc_ob.cptr(),
                       datvec[0].eloc_ob.size(), dat_reduced.eloc_ob.cptr(),
                       std::plus<dcomplex>());
boost::mpi::all_reduce(orz::world(), datvec[0].eloc_ow.cptr(),
                       datvec[0].eloc_ow.size(), dat_reduced.eloc_ow.cptr(),
                       std::plus<dcomplex>());
boost::mpi::all_reduce(orz::world(), datvec[0].eloc_oc.cptr(),
                       datvec[0].eloc_oc.size(), dat_reduced.eloc_oc.cptr(),
                       std::plus<dcomplex>());
boost::mpi::all_reduce(orz::world(), datvec[0].eloc_ox.cptr(),
                       datvec[0].eloc_ox.size(), dat_reduced.eloc_ox.cptr(),
                       std::plus<dcomplex>());
boost::mpi::all_reduce(orz::world(), datvec[0].S_the.cptr(),
                       datvec[0].S_the.size(), dat_reduced.S_the.cptr(),
                       std::plus<double>());
boost::mpi::all_reduce(orz::world(), datvec[0].S_tau.cptr(),
                       datvec[0].S_tau.size(), dat_reduced.S_tau.cptr(),
                       std::plus<double>());
auto &dat = datvec[0];
// compute sampled distribution, tot_times
cout << "nsamples = " << dat.nsamples << endl;
for (size_t i = 0; i < nDets; ++i) {
  dat.prob_of_det.cptr()[i] =
      static_cast<float>(dat.nsamples_of_det.cptr()[i]) / dat.nsamples;
  cout << format("sampled prob: %4d(%20d) %17.10f") % i % bitsvec[i] %
              dat.prob_of_det.cptr()[i]
       << endl;
}
// take the averages ...
{
  const double normf = 1. / dat.nsamples;
  dat.etot *= normf; // total energy
  dat.Ot_b.scale(normf);
  dat.Ot_w.scale(normf);
  dat.Ot_c.scale(normf);
  dat.Ot_x.scale(normf);
  dat.eloc_ob.scale(normf);
  dat.eloc_ow.scale(normf);
  dat.eloc_oc.scale(normf);
  dat.eloc_ox.scale(normf);
  dat.S_the.scale(normf);
  dat.S_tau.scale(normf);
}
// compute total energy with sampled sum
// cout << format("Sampled local energy sum: %20.10f") % dat.etot << endl;
cout << format("Sampled total energy: %24.12f (+nuc) %+24.12f") %
            (dat.etot + base_energy).real() % dat.etot
     << endl;
// evaluate Ot with MH samples
// calculate gradients F for SR
// cout << "−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−" << endl;
if (update_theta) {
  eval_F(N, dat.Ot_b, dat.eloc_ob, dat.etot, dat.F_b);      // for b
  eval_Fwx(N, N, dat.Ot_w, dat.eloc_ow, dat.etot, dat.F_w); // for w
  // Concatenate F_ of theta
  concatvec(N, Ma, dat.F_b, dat.F_w, dat.F_the);
  // Evaluate S_ for theta
  eval_S(N, Ma, dat.Ot_b, dat.Ot_w, dat.size_the, −1.0, dat.S_the, dat.workspc);
  // get S_inverse*F_ for updating theta
  getSinverseF(dat.size_the, dat.F_the, dat.S_the, dat.D_the, 1 e− 4,
               dat.workspc);
} else {
  dat.F_the <<= 0.0;
  dat.D_the <<= 0.0;
}
constexpr std::complex<double> imf(0.0, 1.0);
eval_F(N, dat.Ot_c, dat.eloc_oc, dat.etot *imf, dat.F_c);      // for c
eval_Fwx(N, N, dat.Ot_x, dat.eloc_ox, dat.etot *imf, dat.F_x); // for x
// Concatenate F_ of tau
concatvec(N, Mp, dat.F_c, dat.F_x, dat.F_tau);
// Evaluate S_ for tau
eval_S(N, Mp, dat.Ot_c, dat.Ot_x, dat.size_tau, −1.0, dat.S_tau, dat.workspc);
// get S_inverse*F_ for updating tau
getSinverseF(dat.size_tau, dat.F_tau, dat.S_tau, dat.D_tau, 1 e− 4,
             dat.workspc);
}
// ...... //
// generation of samples (or distributions) based on RBM model with phi(v|theta)
const long niters = 3500 * 120;
// record total occurance of each config for 100 walkers over niters
auto nwalkers = 100 uL;
auto scale_theta = 0.01;
auto update_theta = true; // false;
// Implementation of momentum methods −−−−−−−−−−−−−−−−−−−−−−−−−−
double momentum = 0.5; // initial value
orz::DTensor velocity_b(N);
orz::DTensor velocity_w(N, N);
orz::DTensor velocity_c(N);
orz::DTensor velocity_x(M, N);
// monitor learning
orz::DTensor ratio_b(N);
orz::DTensor ratio_w(N, N);
orz::DTensor ratio_c(N);
orz::DTensor ratio_x(N, N);
// averaging learning
orz::DTensor Sum_the(N + N * N);
orz::DTensor Sum_tau(N + N * N);
int avg_counter = 0;
// MPI
cout << "nthead = " << orz::openmp::nthread() << endl;
const int nproc = orz::world().size();
cout << "nproc = " << nproc << endl;
const int myrank = orz::world().rank();
cout << "myrank = " << myrank << endl;
OptData dat_reduced;
std::vector<OptData> datvec(orz::openmp::nthread());
// Distributing walkers to processes
int extra = nwalkers % nproc;
cout << "extra = nwalkers /% nproc = " << extra << endl;
int proportions = (nwalkers− extra) / nproc;
cout << "proportions = (nwalkers − extra)/nproc = " << proportions << endl;
if (myrank < extra)
  nwalkers = proportions + 1;
else
  nwalkers = proportions;
cout << "nwalkers = " << nwalkers << endl;
// initialize datvec with new walker size
for (long i = 0; i < datvec.size(); ++i)
  datvec[i].init(nDets, nwalkers, N, Ma, Mp, i);
// start of learning
for (size_t iter = 0; iter < 8000; ++iter) {
  orz::ProgressTimer pt("time: update:");
  cout << "\n−−> Opt: " << (static_cast<double>(iter) / 10) << "%" << endl;
  // orz::LoadBin("aaa.bin") >> avec >> bvec >> wmat >> cvec >> dvec >> xmat;
  update_theta = (iter > 100) ? true : false;
  evalgrad_metropolis(niters, nDets, base_energy, bitsvec, bvec, wmat, cvec,
                      xmat, nwalkers, N, Ma, Mp, matH, update_theta, datvec,
                      dat_reduced);
  auto &dat = datvec[0];
  /*
  getDeterministicValues(nDets, base_energy, N, Ma, Mp,
  bvec, wmat,cvec, xmat,
  matH, bitsvec, dat_reduced);
  auto &dat = dat_reduced;
  */
  cout << format(" b=%.5e w=%.5e c=%.5e x=%.5e") % orz::tensor::normf(bvec) %
              orz::tensor::normf(wmat) % orz::tensor::normf(cvec) %
              orz::tensor::normf(xmat)
       << endl;
  cout << "F_the = " << orz::tensor::normf(dat.F_the)
       << " F_tau = " << orz::tensor::normf(dat.F_tau) << endl;
  double normSinverseF_the = orz::tensor::normf(dat.D_the);
  scale_theta =
      (iter > 300)
          ? std::min(0.01, std::max(0.015, 0.05 * 1.0 / normSinverseF_the))
          : 0.018;
  // scale_theta = (iter > 400) ? std::min(0.8, std::max(0.02,
  // 0.02*1.0/normSinverseF_the)) : 0.02; scale_theta =
  // (orz::tensor::normf(datvec[0].F_the) < 0.1) ? std::min(0.8, std::max(0.2,
  // 0.2*1.0/normSinverseF_the)) : 0.02;
  cout << "normSinverseF_the = " << normSinverseF_the << endl;
  cout << "scale_theta = " << scale_theta << endl;
  if (myrank == 0) {
    writebehave(iter, nDets, dat.prob_of_det, bitsvec, dat.etot, base_energy,
                bvec, wmat, cvec, xmat, dat.F_the, dat.F_tau, dat.D_the,
                dat.D_tau, scale_theta);
    monitor_learning(iter, ratio_b, ratio_w, ratio_c, ratio_x);
  }
  // Update tau and theta values
  // when above some iteration, average the updates
  if (iter < 950) {
    updateParam(momentum, iter, N, Ma, bvec, wmat, dat.D_the, scale_theta,
                ratio_b, ratio_w, velocity_b, velocity_w);
    updateParam(momentum, iter, N, Mp, cvec, xmat, dat.D_tau, 0.018, ratio_c,
                ratio_x, velocity_c, velocity_x);
  } else {
    if (avg_counter < 3) {
      for (size_t i = 0; i < dat.size_the; ++i) {
        Sum_the.cptr()[i] = (avg_counter == 0)
                                ? dat.D_the.cptr()[i]
                                : Sum_the.cptr()[i] + dat.D_the.cptr()[i];
        Sum_tau.cptr()[i] = (avg_counter == 0)
                                ? dat.D_tau.cptr()[i]
                                : Sum_the.cptr()[i] + dat.D_the.cptr()[i];
      }
      avg_counter++;
    } else {
      for (size_t i = 0; i < dat.size_the; ++i) {
        Sum_the.cptr()[i] /= 3;
        Sum_tau.cptr()[i] /= 3;
      }
      updateParam(N, Ma, bvec, wmat, Sum_the, scale_theta, ratio_b, ratio_w);
      updateParam(N, Mp, cvec, xmat, Sum_tau, 0.018, ratio_c, ratio_x);
      avg_counter = 0;
    }
  }
}
// ...... //
