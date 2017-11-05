module AD.Reverse exposing (..)

import List exposing (..)
import Dict exposing (Dict)

type Node = Variable String Float
          | Const Float
          | Node Float (List Edge)

-- Edge toNode weight
type Edge = Edge Node Float

--nodeValue : Node -> Float
--nodeValue node =
  --case node of
    --Variable _ v -> v
    --Const v -> v
    --Node v _ -> v

sqr : Node -> Node
sqr n =
  case n of
    Variable label val -> Node (val^2) [Edge n (2*val)]
    Const val -> Const (val^2)
    Node val _ -> Node (val^2) [Edge n (2*val)]

exp : Node -> Node
exp n =
  case n of
    Variable label val -> Node (Basics.e^val) [Edge n (Basics.e^val)]
    Const val -> Const (Basics.e^val)
    Node val _ -> Node (Basics.e^val) [Edge n (Basics.e^val)]

sin : Node -> Node
sin n =
  case n of
    Variable label val -> Node (Basics.sin val) [Edge n (Basics.cos val)]
    Const val -> Const (Basics.sin val)
    Node val _ -> Node (Basics.sin val) [Edge n (Basics.cos val)]

cos : Node -> Node
cos n =
  case n of
    Variable label val -> Node (Basics.cos val) [Edge n (Basics.sin -val)]
    Const val -> Const (Basics.cos val)
    Node val _ -> Node (Basics.cos val) [Edge n (Basics.sin -val)]

pow : Node -> Node -> Node
pow n1 n2 =
  let
      node v1 v2 n1 n2 = Node (v1^v2) [Edge n1 (v2*v1^(v2-1))]
  in
      case (n1,n2) of
        (Variable l1 v1, Const v2) -> node v1 v2 n1 n2
        (Node v1 _, Const v2) -> node v1 v2 n1 n2
        _ -> Debug.crash "unsupported pow operation"

(|^|) = pow


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

(|+|) = add

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

(|.|) = mul
(|*|) = mul

-- graph for f(x,y) = (x^2 + x.y + y^2)
g1 : Node
g1 =
  let
      x = Variable "x" 3
      y = Variable "y" 2
      z = (x |.| x |.| x) |+| (y |.| x) |+| (y |.| y)
  in
      z

-- (a+b)^2 . e^(2.(b+1)) + sin (a+b)^2
g2 : Node
g2 =
  let
      a = Variable "a" 3
      b = Variable "b" 2
      u = (a |+| b) |.| (a |+| b)
      v = (b |+| Const 1)
      w = sqr (exp v)
      z = u |.| w |+| sin u
  in
      z

g3 x y =
  let
      a = Variable "x" x
      b = Variable "y" y
      z = a |.| b |.| a |.| b
  in
      z

g4 x y =
  let
      a = Variable "x" x
      b = Variable "y" y
      z = (a |.| a) |+| (b |.| b)
  in
      z

g5 x = Variable "x" x |^| Const -0.5

g6 x y =
  let
      a = Variable "x" x
      b = Variable "y" y
  in
      cos (pow ((a |.| a) |+| (b |.| b)) (Const 0.5))

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
