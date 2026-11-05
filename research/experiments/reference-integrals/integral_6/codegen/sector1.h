* The name of the loop integral
#define name "integral_6"

* Whether or not we are producing code for contour deformation
#define contourDeformation "0"

* Whether or not complex return type is enforced
#define enforceComplex "0"

* number of integration variables
#define numIV "3"

* number of regulators
#define numReg "1"

#define integrationVariables "x1,x2,x3"
#define realParameters "PP,P1P1,P2P2,M1sq,M2sq,M3sq,R2MID,LL"
#define complexParameters ""
#define regulators "eps"
Symbols `integrationVariables'
        `realParameters'
        `complexParameters'
        `regulators';

#define defaultQmcTransform "korobov3x3"

* Define the imaginary unit in sympy notation.
Symbol I;

#define calIDerivatives "SecDecInternalCalI"
#define functions "`calIDerivatives',SecDecInternalRemainder"
CFunctions `functions';

#define decomposedPolynomialDerivatives "SecDecInternalDecoPoly0"
CFunctions `decomposedPolynomialDerivatives';

* Temporary functions and symbols for replacements in FORM
AutoDeclare CFunctions SecDecInternalfDUMMY;
AutoDeclare Symbols SecDecInternalsDUMMY;

* We generated logs in the subtraction and pack denominators
* and powers into a functions.
CFunctions log, exp, SecDecInternalPow, SecDecInternalDenominator, sqrt;

* We rewrite function calls as symbols
#Do function = {`functions',`decomposedPolynomialDerivatives',log,exp,SecDecInternalPow,SecDecInternalDenominator,sqrt}
  AutoDeclare Symbols SecDecInternal`function'Call;
#EndDo

* We need labels for the code optimization
AutoDeclare Symbols SecDecInternalLabel;

* The integrand may be longer than FORM can read in one go.
* We use python to split the the expression if necessary.
* Define a procedure that defines the "integrand" expression
#procedure defineExpansion
  Global expansion = SecDecInternalsDUMMYIntegrand;
    Id SecDecInternalsDUMMYIntegrand = (( + (( + (1)) * (( + (1))^(-1)))) * ( + (((( + (1))^( + (1))) * (( + ( + (1)))^( + (1)))) * (SecDecInternalCalI( + (1)*x1, + (1)*x2, + (1)*x3, + (0))))));

#endProcedure

#define highestPoles "0"
#define requiredOrders "0"
#define numOrders "1"

* Specify and enumerate all occurring orders in python.
* Define the preprocessor variables
* `shiftedRegulator`regulatorIndex'PowerOrder`shiftedOrderIndex''.
#define shiftedRegulator1PowerOrder1 "0"

* Define two procedures to open and close a nested argument section
#procedure beginArgumentDepth(depth)
  #Do recursiveDepth = 1, `depth'
    Argument;
  #EndDo
#endProcedure
#procedure endArgumentDepth(depth)
  #Do recursiveDepth = 1, `depth'
    EndArgument;
  #EndDo
#endProcedure

* Define procedures to insert the dummy functions introduced in python and their derivatives.
#procedure insertCalI
    Id SecDecInternalCalI(x1?,x2?,x3?,eps?) = (SecDecInternalRemainder( + (1)*x3, + (1)*x2, + (1)*x1, + (1)*eps));

#endProcedure

#procedure insertOther
    Id SecDecInternalRemainder(x1?,x2?,x3?,eps?) = ( + (LL)) * ( + (1)*x1) * ((( + (-1)) + (exp( + (LL)*x3))) ^ ( + (2))) * ( + (-1)*x1 + (1)) * (exp( + (-2*LL)*x3)) * (exp((( + (-1)) * (( + (1)) + (( + (-1)) * (exp( + (-LL)*x3)))) * ( + (P1P1)*x1^2*x2^2 + (R2MID)*x1^2*x2 + (P2P2)*x1^2 + (-2*P1P1)*x1*x2^2 + (-R2MID)*x1*x2 + (P1P1)*x2^2)) + (( + (-1)) * (( + (-1)) + (exp( + (LL)*x3))) * ( + (-P1P1)*x1^2*x2^2 + (P1P1 + P2P2 - PP)*x1^2*x2 + (-P2P2)*x1^2 + (2*P1P1)*x1*x2^2 + (-M2sq + M3sq - 2*P1P1 - P2P2 + PP)*x1*x2 + (M1sq - M3sq + P2P2)*x1 + (-P1P1)*x2^2 + (M2sq - M3sq + P1P1)*x2 + (M3sq)))));

#endProcedure

#procedure insertDecomposed
    Id SecDecInternalDecoPoly0(x1?,x2?,x3?,eps?) =  + ( + (1));

#endProcedure

* Define how deep functions to be inserted are nested.
#define insertionDepth "5"
