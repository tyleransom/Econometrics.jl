function wald(obj::EconometricModel)
    β = coef(obj)
    V = vcov(obj)
	k = length(β) - 1
	rdf = dof_residual(obj)
	R = hasintercept(obj) ? hcat(zeros(k), I) : I
	Bread = R * β
	Meat = inv(bunchkaufman!(Hermitian(R * V * R')))
	W = (Bread' * Meat * Bread) / size(R, 1)
	F = FDist(size(R, 1), rdf)
	p = ccdf(F_Dist, Wald)
	W, F, p
end