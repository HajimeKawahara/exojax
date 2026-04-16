from exojax.database._common.radis_adapter import get_auto_memory_mapping_engine

def _set_engine(engine):
    """
    set engine for radis api

    Args:
        engine (_type_): engine for radis api ("pytables" or "vaex" or None). if None, radis automatically determines.

    Returns:
        str: engine selected
    """
    if engine == None:
        engine_selected = get_auto_memory_mapping_engine()
    else:
        engine_selected = engine
    print("radis engine = ", engine_selected)
    return engine_selected
