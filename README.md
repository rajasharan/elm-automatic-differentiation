## Automatic differentiation in reverse mode for elm

This library calculates the paritial derivatives of a multi-variable function
using the method of automatic differentiation in reverse mode. The result is returned
as a dictionary of keys and their corresponding derivative values (gradient vector).

- a.k.a [algorithmic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- a.k.a [backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

## Usage
### Install
```sh
elm-package install rajasharan/elm-automatic-differentiation
```

### Import
```elm
import Dict exposing (Dict)
import AD.Reverse as AD
  exposing
    ( pow, sqr, exp
    , add, mul
    , (|+|), (|.|), (|*|), (|^|)
    , autodiff
    )
```

### API usage
```elm
-- build a computation graph for your function, for e.g.,
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
result = autodiff (f 3 2)

-- [("x",4044.1999630459845),("y",24215.639637682732)]
-- this means:
-- ∂f/∂x = 4044.19996 at (x=3, y=2)
-- ∂f/∂y = 24215.6396 at (x=3, y=2)
```

#### Another example
```elm
-- g(x) = x^2
g : Float -> AD.Node
g x =
  let
      a = AD.Variable "x" x
  in
      a |^| (AD.Const 2)

result2 = autodiff (g 6)
-- [("x", 12)]
-- ∂g/∂x = 12 at (x=6)
```

## Further Reading
* [Rufflewind's Scratchpad - Reverse-mode automatic differentiation: a tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
* [Daniel Brice - Automatic Differentiation in Haskell](https://www.youtube.com/watch?v=q1DUKEOUoxA)
* [github.com/friedbrice/AutoDiff](https://github.com/friedbrice/AutoDiff)

### [License](/LICENSE)
The MIT License (MIT)
