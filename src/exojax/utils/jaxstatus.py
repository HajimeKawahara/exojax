from jax.config import config
import warnings

def check_jax64bit(allow_32bit):
        """check if the JAX precision mode is 64bit mode

        Args:
            allow_32bit (_type_): exception to use 32bit mode. if True, just send warning message

        Raises:
            ValueError: _description_
        """
        how_change_msg =  "You can change to 64bit mode by writing \n\n"
        how_change_msg += "    from jax import config \n"
        how_change_msg += '    config.update("jax_enable_x64", True)'+"\n"
            
        if not config.values["jax_enable_x64"] and allow_32bit:
            msg = "JAX is 32bit mode. We recommend to use 64bit mode. \n"
            warnings.warn(msg+how_change_msg)
        elif not config.values["jax_enable_x64"]:
            msg = "JAX 32bit mode is not allowed. Use allow_32bit = True or \n"
            raise ValueError(msg+how_change_msg)

