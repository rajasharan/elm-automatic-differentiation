module AD.Reverse
  exposing
    ( Node(Variable, Const)
    , sqr, exp, sin, cos
    , pow, add, mul
    , (|+|), (|.|), (|*|), (|^|)
    , autodiff, nodeValue
    )

{-| This library calculates the paritial derivatives of a multi-variable function
using the method of automatic differentiation in reverse mode. The result is returned
as a dictionary of keys and their corresponding derivative values (a.k.a the gradient vector).

# Definition
@docs Node

# Common helpers for building the computation graph
@docs sqr, exp, sin, cos, pow, add, mul

# Infix operators
@docs (|+|), (|.|), (|*|), (|^|)

# Automatic differentiation
@docs autodiff

# Misc.
@docs nodeValue

-}

import List exposing (..)
import Dict exposing (Dict)

{-| Represent values as a Node in the computation graph. Build up the function
by combining various Variable and Const with the help of helper functions below.

    a = Variable "x" 3.2
    b = Variable "y" 4.7
    c = a |*| (b |+| (Const 1))
-}
type Node
  = Variable String Float
  | Const Float
  | Node Float (List Edge)

-- Edge toNode weight
type Edge = Edge Node Float

{-| Helper method to retrieve the node value
-}
nodeValue : Node -> Float
nodeValue node =
  case node of
    Variable _ v -> v
    Const v -> v
    Node v _ -> v

{-| square a node's computation

    a = Variable "x" 3.2
    b = sqr a
-}
sqr : Node -> Node
sqr n =
  case n of
    Variable label val -> Node (val^2) [Edge n (2*val)]
    Const val -> Const (val^2)
    Node val _ -> Node (val^2) [Edge n (2*val)]

{-| exponent of a node's computation

    a = Variable "x" 3.2
    b = exp a
-}
exp : Node -> Node
exp n =
  case n of
    Variable label val -> Node (Basics.e^val) [Edge n (Basics.e^val)]
    Const val -> Const (Basics.e^val)
    Node val _ -> Node (Basics.e^val) [Edge n (Basics.e^val)]

{-| sin of a node's computation
-}
sin : Node -> Node
sin n =
  case n of
    Variable label val -> Node (Basics.sin val) [Edge n (Basics.cos val)]
    Const val -> Const (Basics.sin val)
    Node val _ -> Node (Basics.sin val) [Edge n (Basics.cos val)]

{-| cos of a node's computation
-}
cos : Node -> Node
cos n =
  case n of
    Variable label val -> Node (Basics.cos val) [Edge n (Basics.sin -val)]
    Const val -> Const (Basics.cos val)
    Node val _ -> Node (Basics.cos val) [Edge n (Basics.sin -val)]

{-| pow of a node's computation. The second arg has to be a constant.
For e.g. to calculate `(x^3)`

    a = Variable "x" 4
    b = pow a (Const 3)
-}
pow : Node -> Node -> Node
pow n1 n2 =
  let
      node v1 v2 n1 n2 = Node (v1^v2) [Edge n1 (v2*v1^(v2-1))]
  in
      case (n1,n2) of
        (Variable l1 v1, Const v2) -> node v1 v2 n1 n2
        (Node v1 _, Const v2) -> node v1 v2 n1 n2
        _ -> Debug.crash "unsupported pow operation"

{-| Infix operator for `pow`
-}
(|^|) : Node -> Node -> Node
(|^|) = pow

{-| Add two node computations
-}
add : Node -> Node -> Node
add n1 n2 =
  let
      node v1 v2 n1 n2 = Node (v1+v2) [Edge n1 1, Edge n2 1]
  in
      case (n1,n2) of
        (Variable l1 v1, Variable l2 v2) -> node v1 v2 n1 n2
        (Const v1, Const v2) -> node v1 v2 n1 n2
        (Node v1 _, Node v2 _) -> node v1 v2 n1 n2
        (Variable l1 v1, Const v2) -> node v1 v2 n1 n2
        (Const v1, Variable l2 v2) -> node v1 v2 n1 n2
        (Node v1 _, Const v2) -> node v1 v2 n1 n2
        (Node v1 _, Variable l2 v2) -> node v1 v2 n1 n2
        (Const v1, Node v2 _) -> node v1 v2 n1 n2
        (Variable l1 v1, Node v2 _) -> node v1 v2 n1 n2

{-| Infix operator for `add`
-}
(|+|) : Node -> Node -> Node
(|+|) = add

{-| Multiply two node computations
-}
mul : Node -> Node -> Node
mul n1 n2 =
  let
      node v1 v2 n1 n2 = Node (v1*v2) [Edge n1 v2, Edge n2 v1]
  in
      case (n1,n2) of
        (Variable l1 v1, Variable l2 v2) -> node v1 v2 n1 n2
        (Const v1, Const v2) -> node v1 v2 n1 n2
        (Node v1 _, Node v2 _) -> node v1 v2 n1 n2
        (Variable l1 v1, Const v2) -> node v1 v2 n1 n2
        (Const v1, Variable l2 v2) -> node v1 v2 n1 n2
        (Node v1 _, Const v2) -> node v1 v2 n1 n2
        (Node v1 _, Variable l2 v2) -> node v1 v2 n1 n2
        (Const v1, Node v2 _) -> node v1 v2 n1 n2
        (Variable l1 v1, Node v2 _) -> node v1 v2 n1 n2

{-| Infix operator for `mul`
-}
(|.|) : Node -> Node -> Node
(|.|) = mul

{-| Infix operator for `mul`
-}
(|*|) : Node -> Node -> Node
(|*|) = mul

traverse : Edge -> Float -> List (String, Float)
traverse edge acc =
  case edge of
    Edge (Variable label val) w -> [(label, acc * w)]
    Edge (Node val edges) w -> concat (map (\e -> traverse e (acc*w)) edges)
    Edge (Const val) w -> []

ad : Node -> Float -> List (String, Float)
ad root acc =
  case root of
    Node v1 edges -> concat (map (\e -> traverse e acc) edges)
    Variable label val -> [(label, acc)]
    Const val -> []

group : List (String , Float) -> Dict String Float
group result =
  let
      dict = Dict.empty
      exists val maybe_val =
        case maybe_val of
          Just v -> Just (v + val)
          Nothing -> Just val
  in
      foldr (\(label, df) d -> Dict.update label (exists df) d) dict result

{-| The `autodiff` function calculates the derivatives of all Variables in a computation.
Returns a dictionary of keys and their corresponding derivative values.

    a = Variable "x" 3
    b = a |^| (Const 2)
    autodiff b == Dict.fromList [("x", 6)]
-}
autodiff : Node -> Dict String Float
autodiff node = group <| ad node 1.0

--z node =
  --case node of
    --Node val _ -> (val, sortBy (\(l,v) -> l) <| Dict.toList <| autodiff node)
    --_ -> Debug.crash "err"

--gradientDescent : (Float -> Float -> Node) -> Float -> Float -> List Float
--gradientDescent fn x y =
  --let
      --iter fn x y n =
        --let cond3 = n < max
            --max = 400000
            --eta = 0.0001
            --min_grad = 0.001
            --df = autodiff (fn x y)
            --dx = Dict.get "x" df |> Maybe.withDefault 0
            --dy = Dict.get "y" df |> Maybe.withDefault 0
            --cond1 = abs dx > min_grad
            --cond2 = abs dy > min_grad
            --val = nodeValue (fn x y)
        --in
            --if (cond1 || cond2) && cond3 then
              --iter fn (x - eta*dx) (y - eta*dx) (n+1)
            --else
              --[ Debug.log "iterations" n
              --, Debug.log "dx" dx
              --, Debug.log "dy" dy
              --, Debug.log "x" x
              --, Debug.log "y" y
              --, Debug.log "val" val
              --]
  --in
      --iter fn x y 0
