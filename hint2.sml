(* Assignment 2. Lazy List *)

datatype 'a lazyList = nullList
          |  cons of 'a * (unit -> 'a lazyList)

val l1 = cons(0, fn () => nullList)
val l2 = cons(1, fn() => l1)
val l3 = cons(2, fn() => l2)
val l4 = cons(3, fn() => l3)

fun make_lazy_list(a, b) =
let val c=b+42
in
    cons(a, fn () => cons(c, fn () => nullList))
end

fun make_lazylist_to_list(ll) = 
    case ll of
     nullList => []
   | cons(a, f) => a::make_list(f())

