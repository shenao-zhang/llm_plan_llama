(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i a e l k j d h c f g b)
(:init 
(handempty)
(ontable i)
(ontable a)
(ontable e)
(ontable l)
(ontable k)
(ontable j)
(ontable d)
(ontable h)
(ontable c)
(ontable f)
(ontable g)
(ontable b)
(clear i)
(clear a)
(clear e)
(clear l)
(clear k)
(clear j)
(clear d)
(clear h)
(clear c)
(clear f)
(clear g)
(clear b)
)
(:goal
(and
(on i a)
(on a e)
(on e l)
(on l k)
(on k j)
(on j d)
(on d h)
(on h c)
(on c f)
(on f g)
(on g b)
)))