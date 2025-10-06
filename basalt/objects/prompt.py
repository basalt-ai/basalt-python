from typing import Dict, Optional, Any, Set
from jinja2 import Template, Environment, meta

from ..ressources.prompts.prompt_types import PromptParams, PromptModel


class Prompt:
    """
    Class representing a prompt in the Basalt system.
    """
    def __init__(self, params: PromptParams):
        self._slug = params.slug
        self._text = params.text
        self._system_text = params.system_text
        self._version = params.version
        self._tag = params.tag
        self._model = params.model
        self._raw_text = params.text
        self._raw_system_text = params.system_text
        self._variables = params.variables

        if params.variables is not None:
            self.compile_variables(params.variables)

    @property
    def slug(self) -> str:
        """Get the prompt slug."""
        return self._slug

    @property
    def text(self) -> str:
        """Get the prompt text."""
        return self._text

    @property
    def system_text(self) -> Optional[str]:
        """Get the prompt system text."""
        return self._system_text

    @property
    def version(self) -> str:
        """Get the prompt version."""
        return self._version

    @property
    def tag(self) -> Optional[str]:
        """Get the prompt tag."""
        return self._tag

    @property
    def model(self) -> PromptModel:
        """Get the prompt model configuration."""
        return self._model

    @property
    def raw_text(self) -> str:
        """Get the original prompt text before variable replacement."""
        return self._raw_text

    @property
    def variables(self) -> Optional[Dict[str, str]]:
        """Get the prompt variables."""
        return self._variables

    @property
    def raw_system_text(self) -> Optional[str]:
        """Get the original system text before variable replacement."""
        return self._raw_system_text

    def compile_variables(self, variables: Dict[str, Any]) -> 'Prompt':
        """Compile the prompt variables."""
        self._variables = variables

        self._text = Template(self._raw_text).render(variables)

        undeclared_variable=self._find_undeclared_variables(self._text)

        if self._raw_system_text:
            self._system_text = Template(self._raw_system_text).render(variables)
            undeclared_variable = undeclared_variable | self._find_undeclared_variables(self._system_text)

        if undeclared_variable:
            print("undeclared variables:", undeclared_variable)

        return self

    @staticmethod
    def _find_undeclared_variables(template: str) -> Set[str]:
        env = Environment()
        ast = env.parse(template)
        variables = meta.find_undeclared_variables(ast)

        return variables

