# OpenAI and structured outputs from OCaml

OpenAI offers a way to have (relatively) typed communication through [structured outputs](https://platform.openai.com/docs/guides/structured-outputs). A json schema must be passed alongside the prompt. And the answer is guaranteed to follow that schema.

In this post I will demonstrate how we can call the OpenAI API from OCaml and use the structured outputs, all with minimal boilerplate thanks to [ppx_deriving_jsonschema](https://github.com/ahrefs/ppx_deriving_jsonschema).

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

In this example, I will demonstrate how to use OpenAI as a math tutor to solve problems with step-by-step details. Instead of receiving a single block of text that would require parsing to extract individual steps, I'll define a structured JSON schema. The schema requires each step to include both an explanation and an intermediate output, along with a final answer. Rather than manually writing the JSON schema as a string or with yojson, I'll define OCaml types using the `[@deriving jsonschema]` annotation. This automatically creates a `TYPENAME_jsonschema` value for each type. Additionally, I use the `[@deriving yojson]` annotation to parse the API output. The `ppx_deriving_jsonschema` package ensures the generated schema is compatible with `ppx_yojson_conv`, `ppx_deriving_yojson`, and `melange-json`.

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
