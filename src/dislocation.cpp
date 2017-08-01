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

    // stress without regularization
    /*double square_i = 1.0 / (pp[0]*pp[0]+pp[1]*pp[1]);
    double square_2i = square_i * square_i;
    tmp[0][0] = -mult * square_2i * pp[1] * (3.0*pp[0]*pp[0] + pp[1]*pp[1]);
    tmp[1][1] = mult * square_2i * pp[1] * (pp[0]*pp[0] - pp[1]*pp[1]);
    tmp[0][1] = mult * square_2i * pp[0] * (pp[0]*pp[0] - pp[1]*pp[1]);
    tmp[1][0] = tmp[0][1];
    */

    // stress with regularization W. Cai
    double a_2 = core_width*core_width;
    double pp0_2 = pp[0]*pp[0];
    double pp1_2 = pp[1]*pp[1];
    double square_i = 1.0 / (pp0_2 + pp1_2 + a_2);

    tmp[0][0] = -mult * square_i * pp[1] * (1+2.0*(pp0_2 + a_2)*square_i);
    tmp[1][1] = mult * square_i * pp[1] * (1-2.0*(pp1_2 + a_2)*square_i);
    tmp[0][1] = mult * square_i * pp[0] * (1-2.0*pp1_2*square_i);
    tmp[1][0] = tmp[0][1];
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

    double square_i = 1.0 / (pp[0]*pp[0]+pp[1]*pp[1]);
    tmp[0] = mult*(0.5*pp[0]*pp[1]*square_i
                              +(1-e.poisson)*std::atan(pp[1]/pp[0]));
    tmp[1] = -0.25*mult*(square_i*(pp[0]*pp[0]-p[1]*p[1])+
                              (1-2.0*e.poisson)*std::log((pp[0]*pp[0]+pp[1]*pp[1])));

    return tmp;
}