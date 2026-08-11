"""The live-UE observation backend: UE renders frames, Python owns the world.

Track A of docs/LIVE_UE_SPEC.md. ``CourierEnv``'s clock is event-driven
bookkeeping -- there is no wall-clock coupling anywhere in the transition
system -- so a UE instance is needed only to *render observations* at poses the
environment fully determines. Everything in this package exists to make a
render service look exactly like a photo album that fills itself in lazily:

``protocol``      the nav-render/v0 wire shapes, pinned by golden fixtures
``client``        one HTTP client per UE instance
``pool``          least-loaded dispatch over many instances, with quarantine
``cache``         the per-episode album directory rendered frames land in
``env``           ``LiveCourierEnv``: CourierEnv with render-on-miss frames
``embodied_env``  Track B: ``EmbodiedCourierEnv``, whose hops a UE pawn walks
                  and whose movement seconds are the engine's own (spec 3b)
``gym_adapter``   the VAGEN/verl training adapters for both backends

See docs/adr/0005-live-ue-observation-backend.md for why this is a subclass
seam rather than a provider refactor or an ``EmbodiedRuntime`` LIVE mode.
"""
