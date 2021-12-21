from functools import partial

from pdoc_to_md import generate_api, format_github_link


def get_icon(module):
    """Get icon of the module."""
    icon = None
    if module.name == 'vectorbtpro.data':
        icon = 'material/database'
    elif module.name == 'vectorbtpro.portfolio':
        icon = 'material/chart-areaspline'
    elif module.name == 'vectorbtpro.utils':
        icon = 'material/tools'
    elif module.name == 'vectorbtpro.generic':
        icon = 'material/matrix'
    elif module.name == 'vectorbtpro.signals':
        icon = 'material/ray-vertex'
    elif module.name == 'vectorbtpro.labels':
        icon = 'material/label-multiple-outline'
    elif module.name == 'vectorbtpro._settings':
        icon = 'material/file-cog'
    elif module.name == 'vectorbtpro.returns':
        icon = 'material/chart-line'
    elif module.name == 'vectorbtpro.base':
        icon = 'material/vector-polyline-plus'
    elif module.name == 'vectorbtpro.indicators':
        icon = 'material/chart-timeline-variant'
    elif module.name == 'vectorbtpro.messaging':
        icon = 'material/message-text-outline'
    elif module.name == 'vectorbtpro.records':
        icon = 'material/table-column'
    elif module.name == 'vectorbtpro.ohlcv':
        icon = 'material/distribute-horizontal-center'
    elif module.name == 'vectorbtpro.registries':
        icon = 'material/text-box-check'
    elif module.name == 'vectorbtpro.px':
        icon = 'material/chart-bubble'
    elif module.name in ('vectorbtpro.utils.schedule_', 'vectorbtpro.data.updater'):
        icon = 'material/timer-outline'
    elif module.fname == 'accessors':
        icon = 'material/arrow-decision'
    elif module.fname == 'factory':
        icon = 'material/factory'
    elif module.fname == 'nb':
        icon = 'material/lambda'
    elif module.fname in ('ca_registry', 'caching'):
        icon = 'material/cached'
    elif module.fname in ('jit_registry', 'jitting'):
        icon = 'material/speedometer'
    elif module.fname in ('ch_registry', 'chunking'):
        icon = 'material/arrow-split-vertical'
    return icon


def get_tags(module):
    """Get tags of the module."""
    module_parts = module.name.split('.')
    tags = set()
    if len(module_parts) > 2:
        for m in module_parts[1:-1]:
            if m == 'nb':
                tags.add('numba')
            else:
                tags.add(m)
    if module.fname == 'accessors':
        tags.add('accessors')
    elif module.fname == 'factory':
        tags.add('factory')
    elif module.fname == 'nb':
        tags.add('numba')
    elif module.fname in ('ca_registry', 'caching'):
        tags.add('caching')
    elif module.fname in ('jit_registry', 'jitting'):
        tags.add('jitting')
    elif module.fname in ('ch_registry', 'chunking'):
        tags.add('chunking')
    return tags


format_github_link = partial(format_github_link, user='polakowo', repo='vectorbt.pro')

if __name__ == "__main__":
    generate_api(
        '../vectorbtpro',
        root_dir='docs',
        get_icon=get_icon,
        get_tags=get_tags,
        format_github_link=format_github_link
    )
