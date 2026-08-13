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
    Vector ComboAdjointMult(real_t a1, real_t a2, real_t dt, Vector &x, real_t t);
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

Vector TopOptRKIMEXSolver::ComboAdjointMult(real_t a1, real_t a2, real_t dt, Vector &x, real_t t)
{
   Vector rhs(f->Width());
   Vector rhs2(f->Width());
   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
   f->AdjointMult(x, rhs2);
   rhs2 *= a1;

   f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
   f->SetTime(t + dt);
   f->AdjointImplicitSolve(dt, x, rhs);
   f->SetTime(t);
   rhs *= a2;
   rhs.Add(1.0, rhs2);
   rhs.Add(1.0, x);
   return rhs;
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

    // adjoint computation 
    f->SetTime(t);
    Vector x_old = x;
    Vector y = x_old; 
    f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
    Vector imp_l(lam.Size());
    Vector exp_l(lam.Size());
    Vector lam_old = lam;
    f->AdjointMult(lam_old, exp_l);

    // state 
    f->Mult(x, k);
    ks_ex.push_back(k);
    x.Add(dt*b_ex(0), ks_ex[0]);

    // design gradient
    Vector exp_grad_placeholder(dJdrho_tilde.Size());
    Vector imp_grad_placeholder(dJdrho_tilde.Size());
    exp_grad_placeholder = 0.0;
    f->ExplicitMultDesignGradient(1.0, lam_old, x_old, exp_grad_placeholder);
    dJdrho_tilde.Add(dt*b_ex(0), exp_grad_placeholder);

    f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
    f->SetTime(t + A_imp(0, 0)*dt);
    f->AdjointImplicitSolve(dt, lam_old, imp_l);
    f->SetTime(t);
    lam.Add(dt*b_ex(0), exp_l);
    for (int stage = 0; stage < num_stages; stage++)
    {
      y = x_old;
      exp_grad_placeholder = 0.0;
      imp_grad_placeholder = 0.0;
      for(int j = stage; j >= 0; j--)
      {
         y.Add(dt*A_ex(stage+1, j), ks_ex[j]);
         if (j != stage){y.Add(dt*A_imp(stage, j), ks_imp[j]);} 
         real_t aimp = (j == stage) ? 0 : A_imp(j, j-1);
         if (j > 0)
         {
            f->ExplicitMultDesignGradient(dt*A_ex(1,0), exp_l, x_old, exp_grad_placeholder);
            dJdrho_tilde.Add(dt*b_ex(stage+1), exp_grad_placeholder);
         }
         f->ExplicitMultDesignGradient(dt*A_ex(1,0), imp_l, x_old, imp_grad_placeholder);
         dJdrho_tilde.Add(dt*b_imp(stage), imp_grad_placeholder);
         exp_l = ComboAdjointMult(dt*A_ex(j+1, j), dt*aimp,  dt*aimp, exp_l, t);
         imp_l = ComboAdjointMult(dt*A_ex(j+1, j), dt*aimp,  dt*aimp, imp_l, t); 
      }
      exp_grad_placeholder = 0.0;
      imp_grad_placeholder = 0.0;
      f->ExplicitMultDesignGradient(dt, lam_old, y, exp_grad_placeholder);
      f->ImplicitSolveDesignGradient(dt, lam_old, y, imp_grad_placeholder);
      dJdrho_tilde.Add(dt*b_ex(stage+1), exp_grad_placeholder);
      dJdrho_tilde.Add(dt*b_imp(stage), imp_grad_placeholder);
      lam.Add(dt*b_ex(stage+1), exp_l);
      lam.Add(dt*b_imp(stage), imp_l);
      if (stage != num_stages - 1)
      {
         f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_1);
         f->AdjointMult(lam_old, exp_l);
         f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
         f->SetTime(t + A_imp(stage+1, stage+1)*dt);
         f->AdjointImplicitSolve(dt, lam_old, imp_l);
         f->SetTime(t);
      }
      f->SetEvalMode(TimeDependentOperator::ADDITIVE_TERM_2);
      f->SetTime(t+A_imp(stage, stage)*dt);
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
