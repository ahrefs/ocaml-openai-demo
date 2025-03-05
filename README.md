# OpenAI and structured outputs from OCaml

## What makes this approach special?

OpenAI offers a way to have (relatively) typed communication through [structured outputs](https://platform.openai.com/docs/guides/structured-outputs). A json schema must be passed alongside the prompt. And the answer is guaranteed to follow that schema.

When working with OpenAI's structured outputs, you typically need to:
1. Define your data structures
2. Create a JSON schema that describes these structures
3. Parse the API responses back into your data structures

This usually involves writing and maintaining JSON schemas by hand, which can be error-prone and tedious. The approach we'll explore in this project eliminates this manual work by:

- Using [ppx_deriving_jsonschema](https://github.com/ahrefs/ppx_deriving_jsonschema) to automatically generate JSON schemas from OCaml types
- Combining it with `ppx_yojson_conv` to handle JSON serialization/deserialization
- Creating a type-safe pipeline from OCaml types to OpenAI API and back

This means you can define your data model once in OCaml and get both the schema for OpenAI and the parsing logic for free!

## Setting up the OpenAI client

The bindings to the API are kept fairly minimal in this example. We are using [Devkit](https://github.com/ahrefs/devkit) for the http client. And `ppx_yojson_conv` to conveniently emit/parse the necessary json.

```ocaml
let api_key = Sys.getenv "OPENAI_API_KEY"

module OpenAI = struct
  let model = "gpt-4o"

  type json = Yojson.Safe.t
  let yojson_of_json x = x

  module Response = struct
    type response_message = { content : string option }
    [@@deriving of_yojson] [@@yojson.allow_extra_fields]

    type choice = { message : response_message }
    [@@deriving of_yojson] [@@yojson.allow_extra_fields]

    type response = { choices : choice list }
    [@@deriving of_yojson] [@@yojson.allow_extra_fields]
  end

  module Request = struct
    type message = {
      role : string;
      content : string;
    }
    [@@deriving yojson_of]

    type json_schema = {
      name : string;
      schema : json;
    }
    [@@deriving yojson_of]

    type response_format = {
      typ : string; [@key "type"]
      json_schema : json_schema;
    }
    [@@deriving yojson_of]

    type request = {
      model : string;
      messages : message list;
      response_format : response_format;
    }
    [@@deriving yojson_of]
  end

  let send ?(debug = false) request =
    let body = `Raw ("application/json", request |> Request.yojson_of_request |> Yojson.Safe.to_string) in
    let () =
      if debug then (
        let (`Raw (_, body)) = body in
        Printf.eprintf "Body: %s\n" body)
    in
    let headers = [ "Authorization: Bearer " ^ api_key ] in
    match Web.http_request ~headers ~body `POST "https://api.openai.com/v1/chat/completions" with
    | `Error e -> Error e
    | `Ok response ->
    try
      let json = Yojson.Safe.from_string response in
      Ok (Response.response_of_yojson json)
    with exn -> Error (Printf.sprintf "error while parsing the response %s: %S" (Printexc.to_string exn) response)
end
```

The types defined in the `Request` and `Response` modules are straight translation of the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create). The `[@@deriving of_yojson]` annotation generates functions to convert JSON from/into these types. The `[@@yojson.allow_extra_fields]` attribute ensures our code won't break if OpenAI adds new fields to their API responses.

The `send` function is a thin wrapper around the HTTP request to OpenAI's API, serializing our request to JSON, sending it, and then parsing the response back into our OCaml types.

Notice how we're using PPX extensions throughout this code to minimize boilerplate. Without these extensions, we would already need to write a lot of manual serialization and deserialization code.

## Creating a structured schema for math problem solving

In this example, I will demonstrate how to use OpenAI as a math tutor to solve problems with step-by-step details. Instead of receiving a single block of text that would require parsing to extract individual steps, I'll define a structured JSON schema. The schema requires each step to include both an explanation and an intermediate output, along with a final answer.

Here's where the magic of `ppx_deriving_jsonschema` comes in:

```ocaml
module Math_reasoning = struct
  type step = {
    explanation : string;
    output : int;
  }
  [@@deriving jsonschema, yojson]

  type math_reasoning = {
    steps : step list;
    final_answer : int;
  }
  [@@deriving jsonschema, yojson] [@@yojson.allow_extra_fields]
end
```

Let's break down what's happening here:

1. We define OCaml types that model our desired response structure
2. The `[@@deriving jsonschema]` annotation automatically generates a JSON schema value called `TYPENAME_jsonschema` for each type
3. The `[@@deriving yojson]` annotation generates functions to convert between our OCaml types and JSON
4. These PPX extensions work together seamlessly - the schema generated by `jsonschema` is compatible with the JSON handling from `yojson`

Without these PPX extensions, we would need to:
- Manually write a JSON schema as a string or build it with yojson
- Write custom code to parse the API responses into our OCaml types
- Ensure the schema and parsing logic stay in sync when our types change

Instead, we get all of this automatically from a single type definition!

## Using the schema in our OpenAI request

Once our schema is ready, it is easy to insert it in the request, alongside with the prompt.

```ocaml
let math_tutor_request user_prompt =
  {
    OpenAI.Request.model = OpenAI.model;
    messages =
      [
        {
          role = "system";
          content =
            "You are a helpful math tutor. You will be provided with a math \
             problem, and your goal will be to output a step by step solution, \
             along with a final answer. For each step, just provide the output \
             as an equation and use the explanation field to detail the \
             reasoning.";
        };
        { role = "user"; content = user_prompt };
      ];
    response_format =
      {
        typ = "json_schema";
        json_schema =
          {
            name = "math_reasoning";
            schema = Math_reasoning.math_reasoning_jsonschema;
          };
      };
  }
```

## Processing the structured response

The remaining task is to retrieve the steps and display them. OpenAI has the ability to return multiple versions of its answer, calling it choices. Here we will only process the first choice for simplicity.

```ocaml
let extract_steps { OpenAI.Response.message = { content }; _ } =
  match content with
  | None -> ()
  | Some content ->
  match content |> Yojson.Safe.from_string |> Math_reasoning.math_reasoning_of_yojson with
  | exception Ppx_yojson_conv_lib.Yojson_conv.Of_yojson_error (exn, json) ->
    Printf.eprintf "unable to parse response, error %s: %s\n" (Printexc.to_string exn) (Yojson.Safe.to_string json)
  | { Math_reasoning.steps; final_answer } ->
    List.iteri
      (fun i { Math_reasoning.explanation; output } ->
        Printf.printf "Step %d: %s\n" i explanation;
        Printf.printf "Output: %f\n" output)
      steps;
    Printf.printf "Final answer: %f\n" final_answer
```

Notice how we're using the `math_reasoning_of_yojson` function that was automatically generated by the `[@@deriving yojson]` annotation. Once we have the properly typed OCaml value, we can safely access its fields and process the steps in a type-safe manner. This is much more robust than manually parsing the JSON or using string manipulation to extract the information.

## Putting it all together

Finally we only have to do a little bit of plumbing to make the program work. We get the question of the user from the command line, query the OpenAI API, and display the response.

```ocaml
let run user_prompt =
  let request = math_tutor_request user_prompt in
  match OpenAI.send request with
  | Error e -> Printf.eprintf "error: %s\n" e
  | Ok { OpenAI.Response.choices = []; _ } -> Printf.eprintf "no choices returned by OpenAI\n"
  | Ok { OpenAI.Response.choices; _ } -> List.iter extract_steps choices

let () =
  let user_prompt = Sys.argv.(1) in
  run user_prompt
```

The output should look like this:

```
$ dune exec ./openai_demo.exe "compute 3+4*17-4/5"
Step 0: First, follow the order of operations, known as PEMDAS (Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right)). Start by handling the multiplication: Calculate 4 * 17 = 68.
Output: 68.000000
Step 1: Next, handle the division: Calculate 4 / 5 = 0.8.
Output: 0.800000
Step 2: Now, handle the addition to and subtraction from the result of the multiplication: Calculate 3 + 68 = 71.
Output: 70.200000
Step 3: Lastly, subtract the result of the division from the addition result: Calculate 71 - 0.8 = 70.2.
Output: 70.200000
Final answer: 70.200000
```

As you can see, the output of some of the steps is not correct. This is expected when performing mathematical operations using an LLM. Please always review the output with care.

The whole project can be found at https://github.com/ahrefs/ocaml-openai-demo.

The dependencies of the project can be installed from opam:

```shell
opam switch create . 5.3.0
opam install dune devkit ppx_yojson_conv ppx_yojson_conv_lib ppx_deriving_jsonschema
```
