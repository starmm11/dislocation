#include "dislocation.hpp"

template <int dim>
SymmetricTensor<2,dim>
EdgeDislocation<dim>::get_stress(const Point<dim>& p,
                            const double& lambda, const double& mu,
                            const double& b)
{
    SymmetricTensor<2, dim> tmp;
    const double nu = 0.5 * lambda / (lambda + mu);
    const double mult = mu * b * 0.5 * PI_i / (1 - nu);
    Tensor<1,dim>  pp = p - coord;
    double square_2i = 1.0 / (pp.norm_square() * pp.norm_square());
    tmp[0][0] = -mult * square_2i * pp[1] * (3.0*pp[0]*pp[0] + pp[1]*pp[1]);
    tmp[1][1] = mult * square_2i * pp[1] * (pp[0]*pp[0] - pp[1]*pp[1]);
    tmp[0][1] = mult * square_2i * pp[0] * (pp[0]*pp[0] - pp[1]*pp[1]);
    tmp[1][0] = tmp[0][1];
    return tmp;
}

template <int dim>
Tensor<1,dim>
EdgeDislocation<dim>::get_u(const Point<dim>& p,
      const double& lambda, const double& mu,
      const double& b)
{
    Tensor<1,dim>  tmp;
    double nu = 0.5 * lambda / (lambda + mu);
    double nu_i = 1.0 / (1-nu);
    Tensor<1,dim>  pp = p - coord;
    if (pp[1]==0 && pp[0] == 0) {
        tmp[0] = 0; tmp[1] = 0;
        return tmp;
    }
    double square_i = 1.0 / (pp[0]*pp[0]+pp[1]*pp[1]);
    tmp[0] = 0.5*b*PI_i*nu_i*(0.5*pp[0]*pp[1]*square_i
                              +(1-nu)*std::atan(pp[1]/pp[0]));
    tmp[1] = 0.5*b*PI_i*nu_i*(0.25*square_i*(pp[1]*pp[1]-p[0]*p[0])-
                              0.25*(1-2.0*nu)*std::log((pp[0]*pp[0]+pp[1]*pp[1])));
    return tmp;
}