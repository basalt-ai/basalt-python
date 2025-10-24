"""Legacy facade module removed in favour of the new Basalt client."""

raise ImportError(
    "basalt.basalt_facade was removed. Use basalt.client.Basalt or the new"
    " PromptsClient/DatasetsClient directly instead."
)
