(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l d g h f c k b)
(:init 
(handempty)
(ontable l)
(ontable d)
(ontable g)
(ontable h)
(ontable f)
(ontable c)
(ontable k)
(ontable b)
(clear l)
(clear d)
(clear g)
(clear h)
(clear f)
(clear c)
(clear k)
(clear b)
)
(:goal
(and
(on l d)
(on d g)
(on g h)
(on h f)
(on f c)
(on c k)
(on k b)
)))