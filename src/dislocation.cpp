#include "dislocation.hpp"
#include <iostream>

template <int dim>
SymmetricTensor<2,dim>
EdgeDislocation<dim>::getStress(const Point<dim> &p,
                                const Elastic &e)
{
    SymmetricTensor<2, dim> tmp;
    static const double emult = e.mu*burgers*0.5*PI_i/(1 - e.poisson);
    double mult = sign*emult;
    Tensor<1,dim>  pp = p - coord;
    std::cout << std::setprecision(8);

    double square_2i = 1.0 / (pp.norm_square() * pp.norm_square());
    tmp[0][0] = -mult * square_2i * pp[1] * (3.0*pp[0]*pp[0] + pp[1]*pp[1]);
    tmp[1][1] = mult * square_2i * pp[1] * (pp[0]*pp[0] - pp[1]*pp[1]);
    tmp[0][1] = mult * square_2i * pp[0] * (pp[0]*pp[0] - pp[1]*pp[1]);
    tmp[1][0] = tmp[0][1];
    //std::cout << "Stress " << tmp[0][0] << ' ' << tmp[1][1] << ' ' << pp[0] << ' ' << pp[1] << '\n';
    return tmp;
}

template <int dim>
Tensor<1,dim>
EdgeDislocation<dim>::getU(const Point<dim> &p,
                           const Elastic &e)
{
    Tensor<1,dim>  tmp;
    static double umult = 0.5*burgers*PI_i/(1-e.poisson);
    double mult = sign*umult;

    Tensor<1,dim>  pp = p - coord;
    if (pp[1] == coord[1] && pp[0] == coord[0]) {
        tmp[0] = 0; tmp[1] = 0;
        return tmp;
    }
    std::cout << std::setprecision(8);

    double square_i = 1.0 / (pp[0]*pp[0]+pp[1]*pp[1]);
    tmp[0] = mult*(0.5*pp[0]*pp[1]*square_i
                              +(1-e.poisson)*std::atan(pp[1]/pp[0]));
    tmp[1] = mult*(0.25*square_i*(pp[1]*pp[1]-p[0]*p[0])-
                              0.25*(1-2.0*e.poisson)*std::log((pp[0]*pp[0]+pp[1]*pp[1])));
    //std::cout << "U " << tmp[0] << ' ' << tmp[1] << ' ' << pp[0] << ' ' << pp[1] << '\n';
    return tmp;
}