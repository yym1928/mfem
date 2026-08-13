#ifndef HT_IMEX_HPP
#define HT_IMEX_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include "HeatTransferSolvers.hpp"
#include "../../pde_filter.hpp"


namespace mfem
{
class TopOptRKIMEXSolver : public ODESolver
{
protected:
    IMEXAdvectionDiffusionSolver *f;
    int num_stages;
    mfem::Array2D<real_t> A_ex; // A_ex must be num_stages+1 x num_stages + 1
    mfem::Array2D<real_t> A_imp;
    Vector b_ex;
    Vector b_imp;  
    std::vector<Vector> ks_ex;
    std::vector<Vector> ks_imp;
    std::vector<Vector> lks_ex;
    std::vector<Vector> lks_imp;
    std::vector<Vector> dks_ex;
    std::vector<Vector> dks_imp;
    std::vector<Vector> x_stages; // for adjoint computation
    Vector k;
    Vector y, yd, yl; // helpers
public:
    void SetButcherTable(mfem::Array2D<real_t> &A_ex_, mfem::Array2D<real_t> &A_imp_, Vector &b_ex_, Vector &b_imp_);
    void Init(IMEXAdvectionDiffusionSolver &f_);
    void AdjointStep(Vector &lam, Vector &x,Vector &dJdrho_tilde, real_t &t, real_t &dt);
    void I_plus_ExplicitMult(Vector &x, Vector &rhs, real_t a);
    void I_plus_ImplicitMult(Vector &x, Vector &rhs, real_t a, real_t dt);
    void Step(Vector &x, real_t &t, real_t &dt);
    static MFEM_EXPORT std::unique_ptr<TopOptRKIMEXSolver> SelectTopOptRKIMEX(const int ode_solver_type);
};

void TopOptRKIMEXSolver::SetButcherTable(mfem::Array2D<real_t> &A_ex_, mfem::Array2D<real_t> &A_imp_, Vector &b_ex_, Vector &b_imp_)
{ 
   A_ex = A_ex_;
   A_imp = A_imp_;
   b_ex = b_ex_;
   b_imp = b_imp_;
}

void TopOptRKIMEXSolver::I_plus_ExplicitMult(Vector &x, Vector &rhs, real_t a)
{
   f->AdjointMult(x, rhs);
   rhs *= a;
   rhs.Add(1.0, x);
}

void TopOptRKIMEXSolver::I_plus_ImplicitMult(Vector &x, Vector &rhs, real_t a, real_t dt)
{
   f->AdjointImplicitSolve(dt, x, rhs);
   rhs *= a;
   rhs.Add(1.0, x);
}


void TopOptRKIMEXSolver::Init(IMEXAdvectionDiffusionSolver &f_)
{
   this->f = &f_;
   mem_type = GetMemoryType(f_.GetMemoryClass());
//    num_stages_ex = A_ex.NumCols();
   // num_stages = A_imp.NumCols();
   int n = f->Width();
   k.SetSize(n, mem_type);
   y.SetSize(n, mem_type);
}


void TopOptRKIMEXSolver::Step(Vector &x, real_t &t, real_t &dt)
{
   f->SetTime(t);
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   f->Mult(x, k);
   Vector x_old = x;
   ks_ex.push_back(k);
   x.Add(dt*b_ex(0), ks_ex[0]);

   for (int stage = 0; stage < num_stages; stage++)
   {
      f->SetTime(t+A_imp(stage, stage)*dt);
      y = x_old;
      for (int j = 0; j < stage; j++)
      {
         y.Add(dt*A_ex(stage+1, j), ks_ex[j]);
         y.Add(dt*A_imp(stage, j), ks_imp[j]); 
      }
      y.Add(dt*A_ex(stage+1, stage), ks_ex[stage]);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->ImplicitSolve(A_imp(stage, stage)*dt, y, k);
      ks_imp.push_back(k);
      y.Add(dt*A_imp(stage, stage), ks_imp[stage]);
      f->SetTime(t);
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
      f->Mult(y, k);
      ks_ex.push_back(k);
      x.Add(dt*b_ex(stage+1), ks_ex[stage+1]);
      x.Add(dt*b_imp(stage), ks_imp[stage]);
   }
   ks_ex.clear();
   ks_imp.clear();
   t += dt;
}

void TopOptRKIMEXSolver::AdjointStep(Vector &lam, Vector &x, Vector &dJdrho_tilde, real_t &t, real_t &dt)
{
    f->SetTime(t);
    f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);

    // state 
    Vector x_old = x;
    f->Mult(x, k);
    x_stages.push_back(x);
    ks_ex.push_back(k);
    x.Add(dt*b_ex(0), ks_ex[0]);

    Vector lam_old = lam;
    Vector dgdrho_tilde(dJdrho_tilde.Size());
    Vector dfdrho_tilde(dJdrho_tilde.Size());
   
    // Compute Adjoint stage. dgdrho_tilde will pop out.
    f->AdjointMult(lam, k);
    lks_ex.push_back(k);
    lam.Add(dt*b_ex(0), k);
   // precompute impl_adj_mult
    Vector impl_adj_mult(lam.Size());
    Vector expl_adj_mult = k;
    f->AdjointImplicitSolve(A_imp(stage, stage)*dt, lam_old, impl_adj_mult);
    f->AdjointMult(impl_adj_mult, k);
    lks_ex.push_back(k);


    // store first gradient term
    dgdrho_tilde = 0.0;
    f->ExplicitMultDesignGradient(1.0, lam_old, x_old, dgdrho_tilde);
    dks_ex.push_back(dgdrho_tilde);
    dJdrho_tilde.Add(dt*b_ex(0), dks_ex[0]);
    yd.SetSize(dgdrho_tilde.Size());
    
    for (int stage = 0; stage < num_stages; stage++)
    {
       f->SetTime(t+A_imp(stage, stage)*dt);
       lks_imp.push_back(k);
       yl = lam_old;
       yd = 0.0;
       y = x_old;
       for (int j = 0; j < stage; j++)
       {
            yl.Add(dt*A_ex(stage+1, j), lks_ex[j]);
            yl.Add(dt*A_imp(stage, j), lks_imp[j]); 
            yd.Add(dt*A_ex(stage+1, j), dks_ex[j]);
            yd.Add(dt*A_imp(stage, j), dks_imp[j]); 
            y.Add(dt*A_ex(stage+1, j), ks_ex[j]);
            y.Add(dt*A_imp(stage, j), ks_imp[j]); 
       }
       y.Add(dt*A_ex(stage+1, stage), ks_ex[stage]);
       //yl.Add(dt*A_ex(stage+1, stage), lks_ex[stage]);
       yd.Add(dt*A_ex(stage+1, stage), dks_ex[stage]);
       f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);

       // state 
       f->ImplicitSolve(A_imp(stage, stage)*dt, y, k);
       ks_imp.push_back(k);
       x.Add(dt*b_imp(stage), ks_imp[stage]);

       // adjoint
       lam.Add(dt*b_imp(stage), yl);
       f->AdjointImplicitSolve(A_imp(stage, stage)*dt, expl_adj_mult, k);
       lks_expl.push_back(k);

       // design gradient computation 
       dfdrho_tilde = 0.0;
       f->ImplicitSolveDesignGradient(A_imp(stage, stage)*dt, lam_old, y, dfdrho_tilde);
       f->ExplicitMultDesignGradient(dt, lks_imp[stage], x_old, dfdrho_tilde);
       dks_imp.push_back(dfdrho_tilde);
       dJdrho_tilde.Add(dt*b_imp(stage), dks_imp[stage]);
       //dJdrho_tilde.Add(dt*b_imp(stage), dfdrho_tilde);

       // explicit part
       y.Add(dt*A_imp(stage, stage), ks_imp[stage]);
       yl.Add(dt*A_imp(stage, stage), lks_imp[stage]);
       yd.Add(dt*A_imp(stage, stage), dks_imp[stage]);
       f->SetTime(t);
       f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1); 
       
       if (b_ex(stage+1) != 0.0)
       {
            // state 
            f->Mult(y, k);
            ks_ex.push_back(k);
            x.Add(dt*b_ex(stage+1), ks_ex[stage+1]); 

            // adjoint
            lam.Add(dt*b_ex(stage+1), yl);
            f->AdjointMult(impl_adj_mult, k);
            lks_ex.push_back(k);

            // design gradient
            dgdrho_tilde = 0.0;
            f->ExplicitMultDesignGradient(1.0, yl, y, dgdrho_tilde);
            dks_ex.push_back(dgdrho_tilde);
            dJdrho_tilde.Add(dt*b_ex(stage+1), dks_ex[stage+1]);
       }
    }
    ks_ex.clear();
    ks_imp.clear();
    lks_ex.clear();
    lks_imp.clear();
    dks_ex.clear();
    dks_imp.clear();
    x_stages.clear();
    t += dt;  
}

class TopOptRKIMEXExpImplEuler : public TopOptRKIMEXSolver
{

   public:
      TopOptRKIMEXExpImplEuler()
      {
         num_stages = 1;
         // mfem::Array2D<real_t> A_ex_EI(num_stages+1, num_stages+1);
         // mfem::Array2D<real_t> A_imp_EI(num_stages, num_stages);
         // Vector b_ex_EI(num_stages+1);
         // Vector b_imp_EI(num_stages);
         A_ex.SetSize(num_stages+1, num_stages+1);
         A_imp.SetSize(num_stages, num_stages);
         b_ex.SetSize(num_stages+1);
         b_imp.SetSize(num_stages);

         A_ex(0,0) = 0.0; A_ex(0,1) = 0.0; A_ex(1,0) = 1.0; A_ex(1,1) = 0.0;
         A_imp(0,0) = 1.0;
         b_ex(0) = 1.0; b_ex(1) = 0.0;
         b_imp(0) = 1.0;
         //SetButcherTable(A_ex_EI, A_imp_EI, b_ex_EI, b_imp_EI);
      }
};

std::unique_ptr<TopOptRKIMEXSolver> TopOptRKIMEXSolver::SelectTopOptRKIMEX(const int ode_solver_type)
{
   using ode_ptr = std::unique_ptr<TopOptRKIMEXSolver>;
   switch (ode_solver_type)
   {
      // L-stable IMEX methods for design opt
      case 1: return ode_ptr(new TopOptRKIMEXExpImplEuler);
      // case 2: return ode_ptr(new TopOptIMEXRK2);

      default: MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type );
   }
}
}
#endif 
