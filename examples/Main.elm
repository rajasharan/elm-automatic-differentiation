module Main exposing (..)

import Dict exposing (Dict)
import AD.Reverse as AD
  exposing
    ( pow, sqr, exp
    , add, mul
    , (|+|), (|.|), (|*|), (|^|)
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
      z = u |*| w |+| AD.sin u
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
      a |^| (AD.Const 2)

result2 = Debug.log "g(x)" <| autodiff (g 6)

-- use elm-repl to play with these functions
-- graph for f(x,y) = (x^2 + x.y + y^2)
g1 : AD.Node
g1 =
  let
      x = AD.Variable "x" 3
      y = AD.Variable "y" 2
      z = (x |.| x |.| x) |+| (y |.| x) |+| (y |.| y)
  in
      z

-- (a+b)^2 . e^(2.(b+1)) + sin (a+b)^2
g2 : AD.Node
g2 =
  let
      a = AD.Variable "a" 3
      b = AD.Variable "b" 2
      u = (a |+| b) |.| (a |+| b)
      v = (b |+| AD.Const 1)
      w = sqr (exp v)
      z = u |.| w |+| AD.sin u
  in
      z

g3 x y =
  let
      a = AD.Variable "x" x
      b = AD.Variable "y" y
      z = a |.| b |.| a |.| b
  in
      z

g4 x y =
  let
      a = AD.Variable "x" x
      b = AD.Variable "y" y
      z = (a |.| a) |+| (b |.| b)
  in
      z

g5 x = AD.Variable "x" x |^| AD.Const -0.5

g6 x y =
  let
      a = AD.Variable "x" x
      b = AD.Variable "y" y
  in
      AD.cos (pow ((a |.| a) |+| (b |.| b)) (AD.Const 0.5))


-- a 2-arg function gradient descent implementation
gradientDescent : (Float -> Float -> AD.Node) -> Float -> Float -> List Float
gradientDescent fn x y =
  let
      iter fn x y n =
        let cond3 = n < max
            max = 400000
            eta = 0.0001
            min_grad = 0.001
            df = autodiff (fn x y)
            dx = Dict.get "x" df |> Maybe.withDefault 0
            dy = Dict.get "y" df |> Maybe.withDefault 0
            cond1 = abs dx > min_grad
            cond2 = abs dy > min_grad
            val = AD.nodeValue (fn x y)
        in
            if (cond1 || cond2) && cond3 then
              iter fn (x - eta*dx) (y - eta*dx) (n+1)
            else
              [ Debug.log "iterations" n
              , Debug.log "dx" dx
              , Debug.log "dy" dy
              , Debug.log "x" x
              , Debug.log "y" y
              , Debug.log "val" val
              ]
  in
      iter fn x y 0

gradients = gradientDescent g6 3 2
