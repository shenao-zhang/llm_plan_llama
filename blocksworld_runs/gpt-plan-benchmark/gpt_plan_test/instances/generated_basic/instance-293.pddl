

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b d)
(on c a)
(on d c)
(clear b)
)
(:goal
(and
(on c d)
(on d b))
)
)


