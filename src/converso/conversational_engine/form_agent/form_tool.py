import json
import operator
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any, Dict, Optional, Type, TypedDict, Union

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field, ValidationError, create_model


class FormToolState(Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"

# We cannot pass directly the BaseModel class as args_schema as pydantic will raise errors,
# so we need to create a dummy class that inherits from BaseModel.


class FormToolInactivePayload(BaseModel):
    pass


class FormToolConfirmPayload(BaseModel):
    confirm: bool = Field(
        description="True if the user confirms the form, False if not or wants to change something."
    )


class FormToolOutcome(BaseModel):
    """
    Represents a form tool output.
    The output is returned as str.
    Any other kwarg is returned in the state_update dict
    """

    output: str
    state_update: Optional[Dict[str, Any]] = None
    return_direct: Optional[bool] = False

    def __init__(
        self,
        output: str,
        return_direct: bool = False,
        **kwargs
    ):
        super().__init__(
            output=output,
            return_direct=return_direct
        )
        self.state_update = kwargs


def make_optional_model(original_model: BaseModel) -> BaseModel:
    """
    Takes a Pydantic model and returns a new model with all attributes optional.
    """
    optional_attributes = {
        attr_name: (
            Union[None, attr_type],
            Field(
                default=None, description=original_model.model_fields[attr_name].description)
        )
        for attr_name, attr_type in original_model.__annotations__.items()
    }

    # Define a custom Pydantic model with optional attributes
    new_class_name = original_model.__name__ + 'Optional'
    OptionalModel = create_model(
        new_class_name,
        **optional_attributes,
        __base__=original_model
    )
    OptionalModel.model_config["validate_assignment"] = True

    return OptionalModel


class FormTool(StructuredTool, ABC):
    form: BaseModel = None
    state: Union[FormToolState | None] = None
    skip_confirm: Optional[bool] = False

    # Backup attributes for handling changes in the state
    args_schema_: Optional[Type[BaseModel]] = None
    description_: Optional[str] = None
    name_: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args_schema_ = None
        self.name_ = None
        self.description_ = None
        self.init_state()

    def init_state(self):
        state_initializer = {
            None: self.enter_inactive_state,
            FormToolState.INACTIVE: self.enter_inactive_state,
            FormToolState.ACTIVE: self.enter_active_state,
            FormToolState.FILLED: self.enter_filled_state
        }
        state_initializer[self.state]()

    def enter_inactive_state(self):
        # Guard so that we don't overwrite the original args_schema if
        # set_inactive_state is called multiple times
        if not self.state == FormToolState.INACTIVE:
            self.state = FormToolState.INACTIVE
            self.name_ = self.name
            self.name = f"{self.name_}Start"
            self.description_ = self.description
            self.description = f"""Starts the form {
                self.name}, which {self.description_}"""
            self.args_schema_ = self.args_schema
            self.args_schema = FormToolInactivePayload

    def enter_active_state(self):
        # if not self.state == FormToolState.ACTIVE:
        self.state = FormToolState.ACTIVE
        self.name = f"{self.name_}Update"
        self.description = f"""Updates data for form {
            self.name}, which {self.description_}"""
        self.args_schema = make_optional_model(self.args_schema_)
        if not self.form:
            self.form = self.args_schema()
        elif isinstance(self.form, str):
            self.form = self.args_schema(**json.loads(self.form))

    def enter_filled_state(self):
        self.state = FormToolState.FILLED
        self.name = f"{self.name_}Finalize"
        self.description = f"""Finalizes form {
            self.name}, which {self.description_}"""
        self.args_schema = make_optional_model(self.args_schema_)
        if not self.form:
            self.form = self.args_schema()
        elif isinstance(self.form, str):
            self.form = self.args_schema(**json.loads(self.form))
        self.args_schema = FormToolConfirmPayload

    def activate(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> FormToolOutcome:
        self.enter_active_state()
        return FormToolOutcome(
            output=f"""Starting form {
                self.name}. If the user as already provided some information, call {self.name}.""",
            active_form_tool=self,
            tool_choice=self.name
        )

    def update(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> FormToolOutcome:
        self._update_form(**kwargs)
        if self.is_form_filled():
            self.enter_filled_state()
            if self.skip_confirm:
                return self.finalize(confirm=True)

            return FormToolOutcome(
                active_form_tool=self,
                output="""Form is filled.
                    Ask the user to confirm the information."""
            )
        else:
            return FormToolOutcome(
                active_form_tool=self,
                output="""Form updated with the provided information.
                    Ask the user for the next field."""
            )

    def finalize(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> FormToolOutcome:
        if kwargs.get("confirm"):
            # The FormTool could use self.form to get the data, but we pass it as kwargs to
            # keep the signature consistent with _run
            result = self._run_when_complete(**self.form.model_dump())
            return FormToolOutcome(
                active_form_tool=None,
                output=result,
                return_direct=self.return_direct
            )

        self.enter_active_state()
        return FormToolOutcome(
            active_form_tool=self,
            output="Ask the user to update the form."
        )

    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        match self.state:
            case FormToolState.INACTIVE:
                return self.activate(*args, **kwargs, run_manager=run_manager)

            case FormToolState.ACTIVE:
                return self.update(*args, **kwargs, run_manager=run_manager)

            case FormToolState.FILLED:
                return self.finalize(*args, **kwargs, run_manager=run_manager)

    @abstractmethod
    def _run_when_complete(self) -> str:
        """
        Should raise an exception if something goes wrong.
        The message should describe the error and will be sent back to the agent to try to fix it.
        """

    def _update_form(self, **kwargs):
        try:
            model_class = type(self.form)
            data = self.form.model_dump()
            data.update(kwargs)
            # Recreate the model with the new data merged to the old one
            # This allows to validate multiple fields at once
            self.form = model_class(**data)
        except ValidationError as e:
            raise ToolException(str(e)) from e

    def get_next_field_to_collect(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        The default implementation returns the first field that is not set.
        """
        if self.state == FormToolState.FILLED:
            return None

        for field_name, field_info in self.args_schema.model_fields.items():
            if not getattr(self.form, field_name):
                return field_name

    def is_form_filled(self) -> bool:
        return self.get_next_field_to_collect() is None

    def get_tool_start_message(self, input: dict) -> str:
        message = ""
        match self.state:
            case FormToolState.INACTIVE:
                message = f"Starting {self.name}"
            case FormToolState.ACTIVE:
                message = f"Updating form for {self.name}"
            case FormToolState.FILLED:
                message = f"Completed {self.name}"
        return message


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: Annotated[Optional[list[BaseMessage]], operator.setitem]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Annotated[Optional[Union[AgentAction,
                                            AgentFinish, None]], operator.setitem]
    # The outcome of a given call to a tool
    # Needs `None` as a valid type, since this is what this will start as
    tool_outcome: Annotated[Optional[Union[FormToolOutcome,
                                           str, None]], operator.setitem]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[Optional[list[tuple[AgentAction,
                                                      FunctionMessage]]], operator.add]
    error: Annotated[Optional[str], operator.setitem]

    active_form_tool: Annotated[Optional[FormTool], operator.setitem]

    # Used to force the agent to call a specific tool
    tool_choice: Annotated[Optional[str], operator.setitem]


class FormReset(BaseTool):
    name = "FormReset"
    description = """Call this tool when the user doesn't want to complete the form anymore. DON'T call it when he wants to change some data."""
    args_schema: Type[BaseModel] = FormToolInactivePayload

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return FormToolOutcome(
            active_form_tool=None,
            output="Form reset. Form cleared. Ask the user what he wants to do next."
        )
