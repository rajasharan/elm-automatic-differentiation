module Main exposing (..)

import Dict exposing (Dict)
import AD.Reverse as AD
  exposing
    ( pow, sqr, exp
    , add, mul, (|+|), (|.|)
    , autodiff
    )

-- e.g usage
-- f(x,y) = (x+y)^2 . e^(2.(y+1)) + sin (x+y)^2
f : Float -> Float -> AD.Node
f x y =
  let
      a = AD.Variable "x" x
      b = AD.Variable "y" y
      u = pow (a |+| b) (AD.Const 2)
      v = (b |+| AD.Const 1)
      w = sqr (exp v)
      z = u |.| w |+| AD.sin u
  in
      z

-- result is a dictionary of keys and their corresponding derivative values
result : Dict String Float
result = Debug.log "f(x,y)" <| autodiff (f 3 2)


-- Another e.g. 
-- g(x) = x^2
g : Float -> AD.Node
g x =
  let
      a = AD.Variable "x" x
  in
      pow a (AD.Const 2)

result2 = Debug.log "g(x)" <| autodiff (g 6)
